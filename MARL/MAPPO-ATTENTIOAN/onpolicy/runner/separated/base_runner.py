import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = self.all_args.num_agents

        print("Num agents: ", self.num_agents)
        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        
        #NEW
        self.num_episodes = self.all_args.num_episodes
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_attention = self.all_args.use_attention
        self.substrate_name= self.all_args.substrate_name
        self.attention_module=self.all_args.attention_module
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir_wandb = str(wandb.run.dir)
                
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                    
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        if self.all_args.algorithm_name == "happo":
            from onpolicy.algorithms.happo.happo_trainer import HAPPO as TrainAlgo
            from onpolicy.algorithms.happo.policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from onpolicy.algorithms.hatrpo.hatrpo_trainer import HATRPO as TrainAlgo
            from onpolicy.algorithms.hatrpo.policy import HATRPO_Policy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            if not self.env_name == "Meltingpot":
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device=self.device)
            else:
                player_key = f"player_{agent_id}"
                rgb_shape = self.envs.observation_space[player_key]["RGB"].shape
                sprite_x = rgb_shape[0]
                sprite_y = rgb_shape[1]

                share_observation_space = self.envs.share_observation_space[player_key] if self.use_centralized_V else \
                    self.envs.share_observation_space[player_key]

                po = Policy(self.all_args,
                            self.envs.observation_space[player_key]['RGB'],
                            share_observation_space,
                            self.envs.action_space[player_key],
                            device=self.device)
                # policy network

            self.policy.append(po)

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            if not self.env_name == "Meltingpot":
                share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                    self.envs.observation_space[agent_id]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[agent_id],
                                           share_observation_space,
                                           self.envs.action_space[agent_id])
            else:
                player_key = f"player_{agent_id}"
                share_observation_space = self.envs.share_observation_space[player_key] if self.use_centralized_V else \
                    self.envs.share_observation_space[player_key]
                bu = SeparatedReplayBuffer(self.all_args,
                                           self.envs.observation_space[player_key]['RGB'],
                                           share_observation_space,
                                           self.envs.action_space[player_key])

            self.buffer.append(bu)
            self.trainer.append(tr)
            
        if self.model_dir is not None and self.all_args.load_model:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            if self.all_args.rnn_attention_module == "LSTM":
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                      self.buffer[agent_id].rnn_states_critic[-1],
                                                                      self.buffer[agent_id].rnn_cells_critic[-1],
                                                                      self.buffer[agent_id].masks[-1])
            else:
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                      self.buffer[agent_id].rnn_states_critic[-1],
                                                                      None,
                                                                      self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)


    def get_wager_objective(self, grad_rewards, comparisson_list, agent_id):
        wager_objective = []
        #comparison_step=3
        
        for i, step_rewards in enumerate(grad_rewards):
            
            # Extract the reward for the specified agent
            agent_reward = step_rewards[agent_id]
            comparisson_rewards = comparisson_list[i][agent_id]
            
            #print("Agent reward: ", agent_reward)
            # Check if the agent reward is greater than the previous three rewards
            if agent_reward > comparisson_rewards:
            #i >= comparison_step and all(agent_reward > grad_rewards[j][agent_id] for j in range(i-comparison_step, i)):
                agent_reward = [agent_reward.item() // 100, 1]
            else:
                agent_reward = [1, agent_reward.item() // 100]
            
            # Append the processed reward to the wager_objective list
            wager_objective.append(agent_reward)
        
        # Fill the rest of the list with zeros until it reaches self.episode_length
        while len(wager_objective) < self.episode_length:
            wager_objective.append([0, 0])
        
        return wager_objective
    
    
    def train(self, episode_length, grad_rewards=None, comparisson_list=None , meta=False):
        train_infos = []
        for agent_id in torch.randperm(self.num_agents):
            wager_objective = self.get_wager_objective(grad_rewards, comparisson_list ,agent_id)

            tmp_buf = self.buffer[agent_id]

            self.trainer[agent_id].prep_training()

            train_info = self.trainer[agent_id].train(tmp_buf, wager_objective=wager_objective ,meta=meta)
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos
    
    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            if self.use_wandb:
                
                torch.save(policy_actor.state_dict(), str(self.save_dir_wandb) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir_wandb) + "/critic_agent" + str(agent_id) + ".pt")
                if self.trainer[agent_id]._use_valuenorm:
                    policy_vnrom = self.trainer[agent_id].value_normalizer
                    torch.save(policy_vnrom.state_dict(), str(self.save_dir_wandb) + "/vnrom_agent" + str(agent_id) + ".pt")
                
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(agent_id) + '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def count_parameters(self):
        actor_parameters = 0
        critic_parameters = 0
        for agent_id in range(self.num_agents):
            actor_parameters += sum(p.numel() for p in self.policy[agent_id].actor.parameters())
            critic_parameters += sum(p.numel() for p in self.policy[agent_id].critic.parameters())
        return actor_parameters + critic_parameters
