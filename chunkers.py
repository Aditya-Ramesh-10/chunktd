import numpy as np
import torch

from curr_utils import normalize_obs


class TrulyTabularModelChunker:
    def __init__(self, 
                 dynamics_model,
                 variant='next_sa'):

        self.dynamics_model = dynamics_model
        self.variant = variant

    def train_model(self, batch_obs, batch_actions, batch_next_obs):
        for obs, action, next_obs in zip(batch_obs, batch_actions, batch_next_obs):
            self.dynamics_model.update(obs, action, next_obs)
        
    def get_next_lambda(self, obs, action, next_obs, next_action, 
                        policy_at_obs, policy_at_next_obs, done):
        if self.variant == 'next_sa':
            return self.get_next_sa_prob(obs, action, next_obs, 
                                         policy_at_next_obs[next_action], done)
        elif self.variant == 'next_s_all_actions':
            return self.get_next_s_prob_all_actions(obs, policy_at_obs, next_obs)
        else:
            raise NotImplementedError

    def get_next_s_prob(self, obs, action, next_obs):
        n_sas = self.dynamics_model.get_nvisits_sas(obs, action, next_obs)
        n_sa = self.dynamics_model.get_nvisits_sa(obs, action)
        return n_sas / n_sa

    def get_next_s_prob_all_actions(self, obs, policy_probs, next_obs):
        n_s = self.dynamics_model.get_nvisits_s(obs)
        next_s_prob = 0
        for a in range(len(policy_probs)):
            n_sas = self.dynamics_model.get_nvisits_sas(obs, a, next_obs)
            next_s_prob += policy_probs[a] * n_sas/n_s
        return next_s_prob

    def get_next_sa_prob(self, obs, action, next_obs, next_action_prob, done):
        p_next_s = self.get_next_s_prob(obs, action, next_obs)
        if done:
            return p_next_s
        p_next_sa = p_next_s * next_action_prob
        return p_next_sa


class CategoricalDeltaDynamicsModelChunker:
    def __init__(self, env, dynamics_model, 
                 lr_dynamics_model, weight_decay_dynamics_model, 
                 replay_buffer, preprocess_obs,
                 preprocess_actions, device, seed):

        self.random_state = np.random.RandomState(seed)
        self.env = env
        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer
        self.device = device

        self.preprocess_obs = preprocess_obs
        self.preprocess_actions = preprocess_actions

        self.dynamics_model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.normalization_tensor = torch.tensor(env.observation_space.high, dtype=torch.float32).to(self.device)

        self.dynamics_model_optimizer = torch.optim.Adam(self.dynamics_model.parameters(),
                                                         lr=lr_dynamics_model,
                                                         weight_decay=weight_decay_dynamics_model)

    def train_model(self, batch_size):

        logs = {}

        batch_obs, batch_actions, batch_next_obs, _, _ = self.replay_buffer.sample(batch_size)

        batch_obs = self.preprocess_obs(batch_obs, device=self.device)
        batch_actions = self.preprocess_actions(batch_actions, num_actions=self.env.action_space.n).to(self.device)
        batch_next_obs = self.preprocess_obs(batch_next_obs, device=self.device)

        model_predicted_next_obs_delta = self.dynamics_model(normalize_obs(batch_obs, self.normalization_tensor), batch_actions)

        target_delta = (batch_next_obs - batch_obs).to(dtype=torch.int64)

        loss = self.criterion(input=model_predicted_next_obs_delta.reshape(-1, self.dynamics_model.max_categories),
                              target=target_delta.reshape(-1))
        
        self.dynamics_model_optimizer.zero_grad()
        loss.backward()
        self.dynamics_model_optimizer.step()

        logs['model_loss'] = loss.item()

        return logs
    
    def add_transition_to_replay_buffer(self, obs, action, next_obs, reward, done):
        self.replay_buffer.add(obs, action, next_obs, reward, done)


    def get_next_sa_prob(self, obs, action, next_obs, next_action_prob, done):
        next_s_prob = self.get_next_s_prob(obs, action, next_obs)
        if done:
            return next_s_prob
        else:
            return next_s_prob * next_action_prob

    def get_next_s_prob(self, obs, action, next_obs):

        obs = self.preprocess_obs([obs], device=self.device)
        action = self.preprocess_actions([action]).to(self.device)
        next_obs = self.preprocess_obs([next_obs], device=self.device)

        with torch.no_grad():
            next_pred_obs_delta = self.dynamics_model(normalize_obs(obs, self.normalization_tensor), action)
            next_pred_obs_delta = torch.nn.functional.softmax(next_pred_obs_delta, dim=-1)

        true_delta = next_obs - obs

        probs = torch.gather(next_pred_obs_delta, 2, true_delta.to(dtype=torch.int64).unsqueeze(-1)).squeeze(-1)
        probs = probs.prod(dim=-1).cpu().numpy()

        assert np.all(probs >= 0) and np.all(probs <= 1)

        return probs


    def get_next_lambda(self, obs, action, next_obs, next_action,
                        policy_at_obs, policy_at_next_obs, done,
                        variant='next_sa'):
        if variant == 'next_sa':
            return self.get_next_sa_prob(obs, action, next_obs, policy_at_next_obs[next_action], done)
        else:
            raise NotImplementedError


class CategoricalDynamicsModelChunker:
    def __init__(self, env, dynamics_model, 
                 lr_dynamics_model,
                 replay_buffer, preprocess_obs,
                 device, seed,
                 use_mask=True):
        
        self.random_state = np.random.RandomState(seed)
        self.env = env
        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer

        self.device = device
        self.preprocess_obs = preprocess_obs

        self.dynamics_model.to(self.device)

        self.normalization_tensor = torch.tensor(env.observation_space.high, dtype=torch.float32).to(self.device)

        self.pred_mask = torch.zeros(self.env.observation_space.shape[0], self.dynamics_model.max_categories, dtype=torch.float32).to(self.device)
        self.pred_mask = self.pred_mask - torch.inf

        for i in range(self.env.observation_space.shape[0]):
            self.pred_mask[i, :int(self.env.observation_space.high[i])+1] = 0

        # print("pred_mask", self.pred_mask.shape)

        self.use_mask = use_mask

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(),
                                          lr=lr_dynamics_model)
        
    def train_model(self, batch_size):

        logs = {}

        batch_obs, batch_actions, batch_next_obs, _, _ = self.replay_buffer.sample(batch_size)

        batch_obs = self.preprocess_obs(batch_obs, device=self.device)

        batch_actions = torch.tensor(batch_actions).to(self.device)
        batch_next_obs = self.preprocess_obs(batch_next_obs, device=self.device)

        model_predicted_next_obs = self.dynamics_model(normalize_obs(batch_obs, self.normalization_tensor), 
                                                       batch_actions)


        if self.use_mask:
            model_predicted_next_obs = model_predicted_next_obs + self.pred_mask

        loss = self.criterion(model_predicted_next_obs.reshape(-1, self.dynamics_model.max_categories),
                              target=batch_next_obs.reshape(-1).to(dtype=torch.int64))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logs['model_loss'] = loss.item()

        return logs
    
    def add_transition_to_replay_buffer(self, obs, action, next_obs, reward, done):
        self.replay_buffer.add(obs, action, next_obs, reward, done)


    def get_next_s_prob_all_actions(self, obs, policy_probs, next_obs, reduce_to_scalar=False):
        obs = self.preprocess_obs(obs, device=self.device)
        next_obs = self.preprocess_obs(next_obs, device=self.device)

        policy_probs = torch.tensor(policy_probs).to(self.device)

        all_actions = torch.arange(self.env.action_space.n, device=self.device)
        rep_obs = obs.repeat_interleave(self.env.action_space.n, dim=0)

        with torch.no_grad():
            next_pred_obs = self.dynamics_model(normalize_obs(rep_obs, self.normalization_tensor),
                                                all_actions)
            
            if self.use_mask:
                next_pred_obs = next_pred_obs + self.pred_mask
            next_pred_obs = torch.nn.functional.softmax(next_pred_obs, dim=-1)

        next_pred_obs = next_pred_obs * policy_probs.unsqueeze(-1).unsqueeze(-1)
        next_pred_obs = next_pred_obs.sum(0)

        next_obs_probs = next_pred_obs[range(next_pred_obs.shape[0]), next_obs.squeeze(0).to(dtype=torch.int64)]

        #print(next_obs_probs)
        if reduce_to_scalar:
            next_obs_prob = next_obs_probs.prod(-1).cpu().numpy()
            l_vector = np.ones(next_obs_probs.shape[-1]) * next_obs_prob
            
        else:
            l_vector = next_obs_probs.cpu().numpy()

        return l_vector
    
    def get_next_lambda(self, obs, action, next_obs, next_action, 
                        policy_at_obs, policy_at_next_obs, done, reduce_to_scalar=False,
                        variant='next_s'):
        if variant == 'next_s':
            # Used for chunked-expected-sarsa
            return self.get_next_s_prob_all_actions([obs], policy_at_obs, [next_obs], reduce_to_scalar=reduce_to_scalar)
        else:
            raise NotImplementedError

