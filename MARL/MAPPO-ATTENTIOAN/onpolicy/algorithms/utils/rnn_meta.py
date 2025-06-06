import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer_Meta(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer_Meta, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        # JUAN AADDED
        # self.encoder = nn.Linear(inputs_dim, outputs_dim)

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
                    
        self.norm = nn.LayerNorm(outputs_dim)
                
        self.wager = nn.Linear(outputs_dim, 2)
                
        self.sigmoid = torch.sigmoid


    def forward(self, x, hxs, masks, prev_h1, prev_h2, cascade_rate1, cascade_rate2, wager=False):
        # JUAN ADDED
        # emb = self.drop(self.encoder(input))
        # emb = emb.to(input.device)
        # if emb.dim()==2:
        #   x = emb.unsqueeze(0)
        # else:
        #    x = emb 

        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (episode_len, batch_num, -1) tensor that has been flatten to (episode_len * batch_num, -1)
            batch_num = hxs.size(0)
            episode_len = int(x.size(0) / batch_num)

            # unflatten
            x = x.view(episode_len, batch_num, x.size(1))

            # Same deal with masks
            masks = masks.view(episode_len, batch_num)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=episode_len to the list
            has_zeros = [0] + has_zeros + [episode_len]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == episode_len
            # x is a (episode_len, batch_num, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(episode_len * batch_num, -1)
            hxs = hxs.transpose(0, 1)

        
        if wager:
            x = self.norm(x)
            
            output_cascade2=x
            if prev_h2 is not None:
                output_cascade2= cascade_rate2*x +  (1-cascade_rate2)*prev_h2
                
            x = self.wager(output_cascade2)
            x = self.sigmoid(x)
            
            return x, hxs, output_cascade2

        else:
            
            output_cascade1=x
            if prev_h1 is not None:
                output_cascade1= cascade_rate1*x +  (1-cascade_rate1)*prev_h1

            x = self.norm(output_cascade1)
                
            return x, hxs, output_cascade1
