import numpy as np

class DecomposedTabularAlgo:

    def __init__(self, observation_space, action_space,
                 update_rule='expected-sarsa-lambda',
                 gamma=0.99, 
                 lr=0.5, 
                 epsilon=0.1,
                 seed=0,
                 ):
        
        self.update_rule = update_rule
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        self.random_state = np.random.RandomState(seed)

        self.reward_components = observation_space.shape[0]

        self.q_tables = np.zeros((self.reward_components,) + tuple(observation_space.high.astype(int)+1) + (action_space.n,))
        self.trace_tables = np.zeros_like(self.q_tables)

        self.default_action_set = np.arange(action_space.n)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_lr(self, lr):
        self.lr = lr

    def _act_randomly(self, state, available_actions=None):
        if available_actions is None:
            available_actions = self.default_action_set
        action = self.random_state.choice(available_actions)
        return action
    
    def _get_overall_q_table(self):
        return np.sum(self.q_tables, axis=0)
    
    def _get_eps_greedy_policy_at_state(self, state, available_actions=None):
        if available_actions is None:
            available_actions = self.default_action_set

        overall_q_table = self._get_overall_q_table()
        state = tuple(state)

        policy = np.zeros_like(overall_q_table[state])
        policy[available_actions] += self.epsilon / len(available_actions)

        q_vals_available_actions = overall_q_table[state][available_actions]

        max_a = np.flatnonzero(np.isclose(q_vals_available_actions, q_vals_available_actions.max(), atol=0))
        for a in max_a:
            policy[available_actions[a]] += (1 - self.epsilon) / len(max_a)

        assert np.isclose(policy.sum(), 1)

        return policy
        
    def _act_greedily(self, state, available_actions):
        if available_actions is None:
            available_actions = self.default_action_set

        overall_q_table = self._get_overall_q_table()
        state = tuple(state)
        a = self.random_state.choice(np.flatnonzero(np.isclose(overall_q_table[state][available_actions], 
                                                               overall_q_table[state][available_actions].max(), 
                                                               atol=0)))
        return a
    
    def act(self, state, available_actions=None):
        if self.random_state.rand() < self.epsilon:
            return self._act_randomly(state, available_actions)
        else:
            return self._act_greedily(state, available_actions)
        

    ##### Online updates: could be made more efficient #####
        
    def update_online(self, state, action, reward_vector, next_state, done, next_action=None, policy_in_next_state=None, lambda_td=None):
        if (self.update_rule == 'chunked-expected-sarsa') or (self.update_rule == 'expected-sarsa-lambda'):
            self._update_variable_expected_sarsa(state, action, reward_vector, next_state, policy_in_next_state, done, lambda_td)
        else:
            raise NotImplementedError
    

    def _update_variable_expected_sarsa(self, state, action, reward_vector, next_state, policy_in_next_state, done, lambda_td_vector):
        state_action = tuple(state) + (action,)
        next_state = tuple(next_state)

        if lambda_td_vector.size == 1:
            self.trace_tables *= self.gamma * lambda_td_vector
        else:
            # decay traces component-wise, could be made more efficient
            for i in range(self.reward_components):
                self.trace_tables[i] *= self.gamma * lambda_td_vector[i]
        # increment traces
        for i in range(self.reward_components):
            self.trace_tables[(i,) + state_action] += 1

        if done:
            td_target = reward_vector
        else:
            td_target = reward_vector + self.gamma * np.array([np.dot(self.q_tables[i][next_state], policy_in_next_state) for i in range(self.reward_components)])

        td_error = td_target - [self.q_tables[(i,) + state_action] for i in range(self.reward_components)]

        for i in range(self.reward_components):
            self.q_tables[i] += self.lr * td_error[i] * self.trace_tables[i]
        
        if done:
            self.trace_tables *= 0