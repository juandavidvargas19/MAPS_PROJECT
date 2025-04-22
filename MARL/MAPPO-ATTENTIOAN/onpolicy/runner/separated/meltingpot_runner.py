import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import cv2
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio
import random
from numpy import array

def _t2n(x):
    return x.detach().cpu().numpy()

def get_episode_parameters(substrate_name):
    parameters = {
        "rationalizable_coordination_in_the_matrix__repeated": (1.0, 1000),
        "factory_commons__either_or": (0.1, 1000),
        "chemistry__two_metabolic_cycles_with_distractors": (0.2, 1000),
        "pure_coordination_in_the_matrix_repeated": (0.1, 1000),
        "prisoners_dilemma_in_the_matrix_repeated": (0.1, 1000),
        "territory_inside_out": (0.2, 1000),
        "commons_harvest__partnership": (0.15, 1000),
        "commons_harvest__closed": (0.15, 1000),
        "predator_prey_open": (1.0, 1000),
        "daycare": (1.0, 1000),
        "allelopathic_harvest__open": (1.0, 2000),
        "territory__rooms": (0.2, 1000),
        "prisoners_dilemma_in_the_matrix__arena": (0.15, 1000),
        "clean_up": (0.2, 1000),
        "territory__open": (0.2, 1000),
        "predator_prey__random_forest": (1.0, 1000),
        "predator_prey__alley_hunt": (1.0, 1000),
        "predator_prey__orchard": (1.0, 1000),
        "chicken_in_the_matrix__arena": (0.2, 1000),
        "chicken_in_the_matrix__repeated": (0.1, 1000),
        "collaborative_cooking__asymmetric": (1.0, 1000),
        "collaborative_cooking__cramped": (1.0, 1000),
        "collaborative_cooking__ring": (1.0, 1000),
        "collaborative_cooking__circuit": (1.0, 1000),
        "collaborative_cooking__forced": (1.0, 1000),
        "collaborative_cooking__figure_eight": (1.0, 1000),
        "collaborative_cooking__crowded": (1.0, 1000),
        "running_with_scissors_in_the_matrix__arena": (0.2, 1000),
        "running_with_scissors_in_the_matrix__repeated": (0.2, 1000),
        "running_with_scissors_in_the_matrix__one_shot": (1.0, 1),
        "stag_hunt_in_the_matrix__arena": (0.1, 1000),            
        "bach_or_stravinsky_in_the_matrix__arena": (0.2, 1000),
        "commons_harvest_open": (0.15, 1000),
        "chemistry__three_metabolic_cycles_with_plentiful_distractors": (0.2, 1000)
    }
    return parameters.get(substrate_name, (0.2, 1000))  # Default values if substrate_name not found


def flatten_lists(input_list):
    # Check if input is a list
    if not isinstance(input_list, list):
        raise ValueError("Input is not a list")

    # Check if each element of the list is also a list
    for inner_list in input_list:
        if not isinstance(inner_list, list):

            # Check if each element of the inner list is a numpy array
            for item in inner_list:
                if not isinstance(item, np.ndarray):
                    raise ValueError("Inner list does not contain only numpy arrays")

    # Convert to a list of concatenated numpy arrays
    concatenated_arrays = [np.concatenate(inner_list) for inner_list in input_list]
    return concatenated_arrays


class MeltingpotRunner(Runner):
    def __init__(self, config):
        super(MeltingpotRunner, self).__init__(config)
        self.config=config

    def run(self):
        self.warmup()

        start = time.time()

        episodes = int(self.num_episodes) 

        print('num episodes to run (separated):', episodes)
        
        print("using meta ", self.all_args.meta, " cascade 1st ", self.all_args.cascade_iterations1, " cascade 2nd ", self.all_args.cascade_iterations2)

        total_num_steps = 0

        for episode in range(episodes):
            self.envs.reset()
            
            
            probability , episode_length = get_episode_parameters(self.substrate_name)
            
            probability = 1.0

            while random.random() > probability and episode_length < self.episode_length:
                episode_length += 100
            print("episode #{} length: {}".format(episode, episode_length) ) 
            
            

            print(f'Episode {episode} start at {time.time()}')
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay_meta(episode, episodes)
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            step_time = time.time()
            rewards = {'player_0': array([0.]), 'player_1': array([0.]), 'player_2': array([0.]), 'player_3': array([0.]), 'player_4': array([0.]), 'player_5': array([0.]), 'player_6': array([0.]), 'player_7': array([0.]), 'player_8': array([0.])}
            
            rewards = np.zeros((self.num_agents, 1))
            
            grad_list=[]
            comparisson_list=[]
            list_rewards = [[] for _ in range(self.num_agents)]
            
            grad_rewards = torch.from_numpy(np.zeros(self.num_agents)).unsqueeze(1)

            for step in range(episode_length):
                
                #rewards old
                rewards_old= rewards
                if isinstance(rewards_old, dict):
                    rewards_old_values = np.array(list(rewards_old.values()))
                elif isinstance(rewards_old, np.ndarray) and step > 0:
                    rewards_old_values=np.array(list(rewards_old[0].values()))
                else:
                    rewards_old_values = rewards_old
                
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env , rnn_states_meta_pre ,rnn_states_critic_meta_pre  = self.collect(step)
                #print("values", values)
                actions = actions.transpose(2, 1, 0)
                #print("actions", actions)
                                          
                obs, rewards, dones, infos = self.envs.step(actions,rewards_old)

                if not isinstance(obs[0], dict):
                    obs = obs[:, 0]
                
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)

                if self.all_args.meta: 
                    if step < episode_length - 1:
                        #meta learning
                        rewards_values=np.array(list(rewards[0].values()))

                        #alpha_values=[0.30 , 0.40 , 0.5]
                        alpha = 0.25 # EMA hyperparameter
                        
                        rewards_values_tensor = []
                        comparisson_values_tensor = []
                        
                        for agent_id in range(self.num_agents):
                            rewards_values_tensor.append((self.buffer[agent_id].rewards)[step] )
                            list_rewards[agent_id].append((self.buffer[agent_id].rewards)[step])
                            comparisson_values_tensor.append( np.mean(  np.array(list_rewards[agent_id] ))   + np.std(np.array((list_rewards[agent_id]))))
                            
                        rewards_values_tensor = torch.from_numpy(np.array(rewards_values_tensor)).squeeze(1)   
                        comparisson_values_tensor = torch.from_numpy(np.array(comparisson_values_tensor))
                        
                        #print(comparisson_values_tensor)                 
                            
                        # EMA calculation
                        grad_rewards = alpha * rewards_values_tensor + (1 - alpha) * grad_rewards   
                        
                        #print("grad_rewards", grad_rewards , grad_rewards.shape)
                        grad_list.append(grad_rewards)
                        comparisson_list.append(comparisson_values_tensor)
                        
                        '''
                        values_meta, actions_meta, action_log_probs_meta, rnn_states_meta, rnn_states_critic_meta, actions_env_meta = self.meta(step, values, actions, action_log_probs, rnn_states_meta_pre, rnn_states_critic_meta_pre, actions_env, grad_rewards)
                        actions_meta = actions_meta.transpose(2, 1, 0)
                        obs_meta, rewards_meta, dones_meta, infos_meta = self.envs.step(actions_meta,rewards_old)
                        
                        #print("step", step, "rewards", rewards)
                        if not isinstance(obs_meta[0], dict):
                            obs_meta = obs_meta[:, 0]
                        
                        #data_meta = obs, rewards_meta, dones_meta, infos, values, actions_meta, action_log_probs_meta, rnn_states, rnn_states_critic
                        #self.insert(data_meta)   
                        '''             


            print(f'Finished {episode_length} steps in {time.time() - step_time} seconds')
            #print(f'Episode {episode} end at {time.time()}')
            # compute return and update network
            self.compute()
            
            train_infos = self.train(episode_length,grad_list, comparisson_list, self.all_args.meta)
            #print(f'Episode {episode} end at {time.time()}')

            #print("finished train infos" ,train_infos)
            # post process
            total_num_steps = total_num_steps + episode_length

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{} total episodes, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_episodes,
                              int(total_num_steps / (end - start))))

                if self.env_name == "Meltingpot":
                    # JUAN ADDED, OVERALL AVERAGE EPISODE REWARDS
                    total_average_episode_rewards = 0
                    
                    #print(self.num_agents, "agents")
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        # Existing code to calculate individual rewards

                        for index in list(self.envs.observation_space.keys())[:self.num_agents]:
                            idv_rews.append(rewards[0][index])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews[:episode_length])})

                        # Calculate the average episode reward for the current agent
                        average_episode_reward = np.mean((self.buffer[agent_id].rewards)[:episode_length]) * episode_length
                        
                        train_infos[agent_id].update({"average_episode_rewards": average_episode_reward})
                        print("Average episode rewards for agent {} is {}".format(agent_id, average_episode_reward))

                        # Add the average reward of this agent to the total
                        total_average_episode_rewards += average_episode_reward

                    # Calculate the overall average episode reward for all agents
                    overall_average_episode_reward = total_average_episode_rewards / self.num_agents
                    print("Overall average episode reward for all agents:", overall_average_episode_reward)

                self.log_train(train_infos, total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        # if --n_rollout_threads 6 && --substrate_name "territory__rooms"
        obs = self.envs.reset()
        # replay buffer
        share_obs = []
        agent_obs = []
        for sublist in obs:
            for item in sublist:
                if item:
                    rgb_player = []
                    arrays = []
                    for agent_id in range(self.num_agents):
                        player = f"player_{agent_id}"
                        if player in item:
                            rgb_player.append(item[player]['WORLD.RGB'])
                            arrays.append(item[player]['RGB'])
                            # player_i obs: (11, 11, 3), share_obs: (168, 168, 3)
                    result = np.stack(arrays)
                    image = np.stack(rgb_player)
            share_obs.append(image)
            agent_obs.append(result)
        share_obs = np.array(share_obs)
        agent_obs = np.array(agent_obs)
        # share_obs shape: (6, 9, 168, 168, 3), agent obs shape: (6, 9, 11, 11, 3)
        for agent_id in range(self.num_agents):
            # size of buffer share_obs (6, 168, 168, 3)--- obs (6, 11, 11, 3)
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id, :, :, :].transpose(0, 2, 1, 3).copy()
            self.buffer[agent_id].obs[0] = agent_obs[:, agent_id, :, :, :].copy()


    def meta(self, step, values_pre, actions_pre, action_log_probs_pre, rnn_states_meta, rnn_states_critic_meta, actions_env_pre, grad_rewards):
        
        #print(rnn_states_meta.shape, rnn_states_critic_meta.shape, grad_rewards.shape)    
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            wager_objective = 1 if grad_rewards.squeeze(1)[agent_id] > 0 else 0

            self.trainer[agent_id].prep_rollout()
            rnn_state_meta= torch.from_numpy(rnn_states_meta).squeeze(0)[agent_id].unsqueeze(0).cpu()
            rnn_state_critic_meta= torch.from_numpy(rnn_states_critic_meta).squeeze(0)[agent_id].unsqueeze(0).cpu()

            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.meta(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            rnn_state_meta,
                                                            rnn_state_critic_meta,
                                                            self.buffer[agent_id].masks[step], 
                                                            wager_objective)
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            player = f"player_{agent_id}"

            if self.envs.action_space[player].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[player].shape):
                    uc_action_env = np.eye(self.envs.action_space[player].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[player].__class__.__name__ == 'Discrete':
                var = np.eye(self.envs.action_space[player].n)[action]
                action_env = np.squeeze(var,
                                        axis=next((axis for axis, size in enumerate(var.shape) if size == 1), None))

            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state[0]))
            rnn_states_critic.append(_t2n(rnn_state_critic[0]))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values) if isinstance(values, list) else values
        actions = np.array(actions) if isinstance(actions, list) else actions
        action_log_probs = np.array(action_log_probs) if isinstance(action_log_probs, list) else action_log_probs
        rnn_states = np.array(rnn_states) if isinstance(rnn_states, list) else rnn_states
        rnn_states_critic = np.array(rnn_states_critic) if isinstance(rnn_states_critic, list) else rnn_states_critic

        #print("all sizes", values.shape, actions.shape , action_log_probs.shape , rnn_states.shape , rnn_states_critic.shape )
        if self.attention_module=="SCOFF":
            values = values.squeeze(2)
            actions = actions.squeeze(2)
            action_log_probs = action_log_probs.squeeze(2)
            rnn_states = rnn_states.squeeze(2)
            rnn_states_critic = rnn_states_critic.squeeze(2)
            
        values = values.squeeze(-1).transpose(1, 0, 2)
        if actions.ndim == 3:
            actions = actions.transpose(2, 0, 1)
        else:
            actions = actions.squeeze(-1).transpose(1, 0, 2)
        if action_log_probs.ndim == 2:
            action_log_probs = action_log_probs[:, np.newaxis, :]
        action_log_probs = action_log_probs.transpose(1, 0, 2)
        if rnn_states.ndim == 3:
            rnn_states = rnn_states[:, np.newaxis, :, :]
        rnn_states = rnn_states.transpose(1, 0, 2, 3)
        if rnn_states_critic.ndim == 3:
            rnn_states_critic = rnn_states_critic[:, np.newaxis, :, :]
        rnn_states_critic = rnn_states_critic.transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
    
    
    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        rnn_states_meta=[]
        rnn_states_critic_meta=[]
        
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            #print(self.buffer[agent_id].rnn_states[step].shape, "rnn_states shape")
            rnn_state_pre=self.buffer[agent_id].rnn_states[step]
            rnn_state_critic_pre=self.buffer[agent_id].rnn_states_critic[step]
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            rnn_state_meta= _t2n(rnn_state[0]) - rnn_state_pre
            rnn_state_critic_meta=_t2n(rnn_state_critic[0]) - rnn_state_critic_pre
            #print(rnn_state.shape)
            
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            player = f"player_{agent_id}"

            if self.envs.action_space[player].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[player].shape):
                    uc_action_env = np.eye(self.envs.action_space[player].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[player].__class__.__name__ == 'Discrete':
                var = np.eye(self.envs.action_space[player].n)[action]
                action_env = np.squeeze(var,
                                        axis=next((axis for axis, size in enumerate(var.shape) if size == 1), None))

            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state[0]))
            rnn_states_critic.append(_t2n(rnn_state_critic[0]))
            
            rnn_states_meta.append(rnn_state_meta)
            rnn_states_critic_meta.append(rnn_state_critic_meta)
            
            

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values) if isinstance(values, list) else values
        actions = np.array(actions) if isinstance(actions, list) else actions
        action_log_probs = np.array(action_log_probs) if isinstance(action_log_probs, list) else action_log_probs
        rnn_states = np.array(rnn_states) if isinstance(rnn_states, list) else rnn_states
        rnn_states_critic = np.array(rnn_states_critic) if isinstance(rnn_states_critic, list) else rnn_states_critic

        rnn_states_meta = np.array(rnn_states_meta) if isinstance(rnn_states_meta, list) else rnn_states_meta
        rnn_states_critic_meta = np.array(rnn_states_critic_meta) if isinstance(rnn_states_critic_meta, list) else rnn_states_critic_meta


        #print("all sizes", values.shape, actions.shape , action_log_probs.shape , rnn_states.shape , rnn_states_critic.shape )
        if self.attention_module=="SCOFF":
            values = values.squeeze(2)
            actions = actions.squeeze(2)
            action_log_probs = action_log_probs.squeeze(2)
            rnn_states = rnn_states.squeeze(2)
            rnn_states_critic = rnn_states_critic.squeeze(2)
            rnn_states_critic_meta=rnn_states_critic_meta.squeeze(2)
            rnn_states_meta=rnn_states_meta.squeeze(2)
            
        values = values.squeeze(-1).transpose(1, 0, 2)
        if actions.ndim == 3:
            actions = actions.transpose(2, 0, 1)
        else:
            actions = actions.squeeze(-1).transpose(1, 0, 2)
        if action_log_probs.ndim == 2:
            action_log_probs = action_log_probs[:, np.newaxis, :]
        action_log_probs = action_log_probs.transpose(1, 0, 2)
        if rnn_states.ndim == 3:
            rnn_states = rnn_states[:, np.newaxis, :, :]
        rnn_states = rnn_states.transpose(1, 0, 2, 3)
        if rnn_states_critic.ndim == 3:
            rnn_states_critic = rnn_states_critic[:, np.newaxis, :, :]
        rnn_states_critic = rnn_states_critic.transpose(1, 0, 2, 3)


        #meta
        if rnn_states_meta.ndim == 3:
            rnn_states_meta = rnn_states_meta[:, np.newaxis, :, :]
        rnn_states_meta = rnn_states_meta.transpose(1, 0, 2, 3)
        if rnn_states_critic_meta.ndim == 3:
            rnn_states_critic_meta = rnn_states_critic_meta[:, np.newaxis, :, :]
        rnn_states_critic_meta = rnn_states_critic_meta.transpose(1, 0, 2, 3)
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env , rnn_states_meta ,rnn_states_critic_meta 

    def process_obs(self, obs):
        """
        This function takes a dict of agents, each agent should be (thread, obs_shape)
        or
        takes a list of dict of agents, each agent should be (obs_shape)
        """
        # todo, check if we need to swap axis
        # Initialize lists to store processed observations
        share_obs = []
        agent_obs = []

        if isinstance(obs, dict):
            # If sublist is a dictionary, process it directly
            for agent_id in range(self.num_agents):
                player = f"player_{agent_id}"
                if player in obs:
                    share_obs.append(obs[player]['WORLD.RGB'])
                    agent_obs.append(obs[player]['RGB'])

            share_obs = np.array(share_obs)
            agent_obs = np.array(agent_obs)
            share_obs = share_obs.transpose(1, 0, 3, 2, 4)
            agent_obs = agent_obs.transpose(1, 0, 3, 2, 4)
        elif isinstance(obs, np.ndarray) or isinstance(obs, list):
            for thread_list in obs:
                per_thread_share = []
                per_thread_obs = []
                for agent_id in range(self.num_agents):
                    player = f"player_{agent_id}"
                    if player in thread_list:
                        share = thread_list[player]['WORLD.RGB']
                        obs = thread_list[player]['RGB']
                        if len(share.shape) > 3:
                            per_thread_share.append(np.squeeze(share, axis=0))
                        else:
                            per_thread_share.append(share)

                        if len(obs.shape) > 3:
                            per_thread_obs.append(np.squeeze(obs, axis=0))
                        else:
                            per_thread_obs.append(obs)

                share_obs.append(per_thread_share)
                agent_obs.append(per_thread_obs)

            share_obs = np.array(share_obs)
            agent_obs = np.array(agent_obs)
            if len(share_obs.shape) != 5:
                print(f'Wrong dim! share obs has shape {share_obs.shape}, obs has shape {agent_obs.shape}')
            share_obs = share_obs.transpose(0, 1, 3, 2, 4)
            agent_obs = agent_obs.transpose(0, 1, 3, 2, 4)
        else:
            print("Error: Obs not in correct data structure !")

        return share_obs, agent_obs

    def insert(self, data):
        obs, rewards, done, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        done_new = self.extract_data(done, np.bool_)
        rewards = self.extract_data(rewards, np.float32)

        # Create a boolean mask with the same shape as rnn_states
        rnn_states[done_new == True] = np.zeros(((done_new == True).sum(), self.hidden_size), dtype=np.float32)
        rnn_states_critic[done_new == True] = np.zeros(((done_new == True).sum(), self.hidden_size), dtype=np.float32)

        masks = np.ones((1, self.num_agents, self.n_rollout_threads, 1), dtype=np.float32)
        masks[done_new == True] = np.zeros(((done_new == True).sum(), 1), dtype=np.float32)

        share_obs, agent_obs = self.process_obs(obs)

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id],
                                         agent_obs[:, agent_id],
                                         rnn_states[:, agent_id].swapaxes(1, 0),
                                         rnn_states_critic[:, agent_id].swapaxes(1, 0),
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id].swapaxes(1, 0),
                                         values[:, agent_id].swapaxes(1, 0),
                                         rewards[:, agent_id].swapaxes(1, 0),
                                         masks[:, agent_id])

    def extract_data(self, original_data, data_type):
        """
        Convert dict into list
        """
        new_data = []
        for per_thread_list in original_data:
            per_thread = []
            for i in range(self.num_agents):
                player_name = f'player_{i}'
                per_thread.append(per_thread_list[player_name])
            new_data.append(per_thread)
        return np.array(new_data, dtype=data_type).transpose(2, 1, 0)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action

                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            # obs = self.envs.reset()
            obs = self.envs.reset()[:, 0]
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array', has_mode=False)[0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    player = f"player_{agent_id}"
                    self.trainer[agent_id].prep_rollout()
                    rgb_data = obs[0][player]['RGB'] if isinstance(obs, np.ndarray) else obs[player]['RGB']
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(np.expand_dims(rgb_data, axis=0))),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True)
                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[player].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[player].shape):
                            uc_action_env = np.eye(self.envs.action_space[player].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[player].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[player].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    if isinstance(rnn_state, tuple):
                        rnn_states[:, agent_id] = _t2n(rnn_state[0])
                    else:
                        rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                actions_env = np.array(actions_env[0]).swapaxes(0, 1)  # Armin
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # Armin: start
                obs = obs[0]

                rewards = rewards[0]
                summed_rewards = []
                for agent_id in range(self.num_agents):
                    summed_rewards.append(np.sum(rewards[player]))
                summed_rewards = np.array(summed_rewards)
                # Armin: end
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                # rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                # masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    # image = self.envs.render('rgb_array')[0][0]
                    image = self.envs.render('rgb_array', has_mode=False)[0]
                    all_frames.append(image)
                    # imageio.imwrite(f"{self.gif_dir}/step_{episode}_{step}_agent.png", image)  # Save the image with agent number appended to filename
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
            player_rewards = {f"player_{agent_id}": [] for agent_id in range(self.num_agents)}

            # Accumulate rewards for each player
            for episode in episode_rewards:
                for agent_id in range(self.num_agents):
                    player = f"player_{agent_id}"
                    player_rewards[player].append(episode[player])

            # Calculate the average reward for each player
            average_episode_rewards = {}
            for agent_id in range(self.num_agents):
                player = f"player_{agent_id}"
                total_rewards = np.sum(player_rewards[player], axis=0)
                average_episode_rewards[player] = np.mean(total_rewards)

            # Print the average rewards
            for player, avg_reward in average_episode_rewards.items():
                print(f"eval average episode rewards of {player}: {avg_reward}")

            # episode_rewards = np.array(episode_rewards)
            # for agent_id in range(self.num_agents):
            #    average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
            #    print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
