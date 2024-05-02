import numpy as np
import torch

def basic_preprocess_obs(obs_list, device):
    np_obs = np.array(obs_list)
    return torch.tensor(np_obs, dtype=torch.float32, device=device)

def normalize_obs(obs_tensor, normalization_tensor):
    return obs_tensor/normalization_tensor

def preprocess_actions(actions_list, num_actions=2, device='cpu'):
    # convert actions to one-hot encoding
    actions = np.array(actions_list)
    one_hot_actions = np.zeros((len(actions_list), num_actions))
    one_hot_actions[np.arange(len(actions_list)), actions] = 1
    return torch.tensor(one_hot_actions, dtype=torch.float32, device=device)