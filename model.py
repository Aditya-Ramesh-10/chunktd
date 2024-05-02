import torch
import torch.nn as nn
import numpy as np

class TrulyTabularModel:
    def __init__(self, obs_space, action_space):
        self.count_sas_table = np.zeros(tuple(obs_space.high.astype(int)+1) + (action_space.n,) + tuple(obs_space.high.astype(int)+1))

    def update(self, obs, action, next_obs):
        self.count_sas_table[tuple(obs.astype(int))][action][tuple(next_obs.astype(int))] += 1

    def get_nvisits_s(self, obs):
        return self.count_sas_table[tuple(obs.astype(int))].sum()

    def get_nvisits_sa(self, obs, action):
        return self.count_sas_table[tuple(obs.astype(int))][action].sum()

    def get_nvisits_sas(self, obs, action, next_obs):
        return self.count_sas_table[tuple(obs.astype(int))][action][tuple(next_obs.astype(int))]

    
class CategoricalStateDeltaDynamicsModel(nn.Module):
    def __init__(self, obs_space, action_space,
                 max_categories, hidden_dim_size=256):
        super().__init__()

        self.obs_dim = obs_space.shape[0]
        self.max_categories = max_categories
        self.action_dim = action_space.n

        self.s_net = nn.Sequential(nn.Linear(self.obs_dim + self.action_dim,
                                           hidden_dim_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim_size,
                                           hidden_dim_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim_size,
                                           hidden_dim_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim_size,
                                           self.obs_dim * self.max_categories))

    def forward(self, obs, action):

        x_i = obs
        x = x_i.reshape(x_i.shape[0], -1)
        x = torch.cat([x, action], dim=1)
        o = self.s_net(x)

        return o.reshape(o.shape[0], self.obs_dim, self.max_categories)
    
   
class CategoricalStateDynamicsModel(nn.Module):
    def __init__(self, obs_space, action_space,
                 action_embedding_dim=8,
                 hidden_dim_size=128):
        super().__init__()

        self.obs_dim = obs_space.shape[0]
        self.action_dim = action_space.n
        self.action_embedding_dim = action_embedding_dim

        self.action_embedding = nn.Embedding(self.action_dim, self.action_embedding_dim)

        self.max_categories = int(obs_space.high.max()) + 1
        self.op_dims = int(self.obs_dim * self.max_categories)

        # predicts categories for each component of the observation
        self.next_s_net = nn.Sequential(nn.Linear(self.obs_dim + self.action_embedding_dim,
                                           hidden_dim_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim_size,
                                           hidden_dim_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_dim_size,
                                           self.op_dims),
                                 )
        
    def forward(self, obs, action):

        x_i = obs
        x = x_i.reshape(x_i.shape[0], -1)

        action = self.action_embedding(action)

        x = torch.cat([x, action], dim=1)
        x = self.next_s_net(x)

        return x.reshape(x.shape[0], self.obs_dim, self.max_categories)