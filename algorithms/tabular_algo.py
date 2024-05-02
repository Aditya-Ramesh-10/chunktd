import numpy as np


class TabularQFunctionBinaryStates:
    def __init__(self, env, gamma, lr, epsilon=0.1,
                 update_rule='sarsa-lambda', seed=0):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_rule = update_rule

        self.q_table = np.zeros(tuple(env.observation_space.high.astype(int)+1) + (env.action_space.n,))
        
        self.state_action_counter = np.zeros_like(self.q_table)
        self.traces_table = np.zeros_like(self.q_table)
        self.lr = lr

        self.random_state = np.random.RandomState(seed)

        self.default_action_set = np.arange(env.action_space.n)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def set_lr(self, lr):
        self.lr = lr

    def _act_randomly(self, state, available_actions):
        return self.random_state.choice(available_actions)
    
    def _act_greedily(self, state, available_actions):
        # Break ties randomly
        state = tuple(state)
        a = self.random_state.choice(np.flatnonzero(np.isclose(self.q_table[state][available_actions], self.q_table[state][available_actions].max(), atol=0)))
        return available_actions[a] 

    def act(self, state, available_actions):
        if self.random_state.rand() < self.epsilon:
            return self._act_randomly(state, available_actions)
        else:
            return self._act_greedily(state, available_actions)

    
    def update_online(self, state, action, reward, next_state, done, next_action=None, td_lambda=None,
                      policy_in_next_state=None):
 
        if (self.update_rule == 'chunked-sarsa') or (self.update_rule == 'sarsa-lambda'):
            self._update_variable_sarsa(state, action, reward, next_state, next_action, done, td_lambda)

        elif (self.update_rule == 'chunked-expected-sarsa') or (self.update_rule == 'expected-sarsa-lambda'):
            self._update_variable_expected_sarsa(state, action, reward, next_state, policy_in_next_state, done, td_lambda)
 
        else:
            raise ValueError('Unknown online update rule')


    def _update_variable_sarsa(self, state, action, reward, next_state, next_action, done, td_lambda):

        state_action = tuple(state) + (action,)
        next_state_action = tuple(next_state) + (next_action,)
        self.traces_table = self.gamma * td_lambda * self.traces_table
        self.traces_table[state_action] += 1
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_table[next_state_action]

        td_error = td_target - self.q_table[state_action]
        self.q_table += self.lr * td_error * self.traces_table
        if done:
            self.traces_table = self.traces_table * 0

    def _update_variable_expected_sarsa(self, state, action, reward, next_state, policy_in_next_state, done, td_lambda):
        state_action = tuple(state) + (action,)
        next_state = tuple(next_state)

        self.traces_table = self.gamma * td_lambda * self.traces_table
        self.traces_table[state_action] += 1
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.sum(policy_in_next_state * self.q_table[next_state])

        td_error = td_target - self.q_table[state_action]
        self.q_table += self.lr * td_error * self.traces_table
        if done:
            self.traces_table = self.traces_table * 0

            
    def _get_eps_greedy_policy_in_state(self, state, available_actions=None):

        if available_actions is None:
            available_actions = self.default_action_set

        state = tuple(state.astype(int))
        policy = np.zeros(len(self.q_table[state]))

        policy[available_actions] += self.epsilon / len(available_actions)

        qvals_available_actions = self.q_table[state][available_actions]

        max_a = np.flatnonzero(np.isclose(qvals_available_actions, qvals_available_actions.max(), atol=0))
        for a in max_a:
            policy[available_actions[a]] += (1 - self.epsilon) / len(max_a)

        assert np.isclose(policy.sum(), 1.0)

        return policy
