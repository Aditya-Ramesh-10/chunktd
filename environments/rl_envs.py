import numpy as np
import gymnasium as gym


class AccumulatedChargeChoice(gym.Env):

    def __init__(self, horizon=100, p_c=0.2, acc_points=10, 
                 bonus=0.1, seed=0,
                 avg_window_size=750):
        super().__init__()

        self.horizon = horizon
        self.p_c = p_c
        self.baseline_charged = 0.5
        self.bonus = bonus

        self.random_state = np.random.RandomState(seed)

        self.plus_indicator = None # set on reset
        self.time = None # set on reset
        self.charged_visits = 0
        self.c_0 = None # set on reset
        self.total_episodes = 0
        self.intermediate_charged_visits = 0
        self.presampled_accumulations = None # set on reset
        self.acc_pointer = None

        self.action_space = gym.spaces.Discrete(self.get_n_actions()[0])

        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]),
                                                high=np.array([1, self.horizon, self.horizon+1]),
                                                shape=(3,))
        
        self.render_mode = None
        self.acc_points = self.random_state.choice(np.arange(1, self.horizon), 
                                                   size=acc_points, replace=False)

        self.acc_points.sort()

        self.avg_window_size = avg_window_size
        self.optimal_choices = np.zeros(self.avg_window_size)
        self.count_sampling_steps = 0
        self.regretful_choices = 0

    def get_n_actions(self):
        return [2]

    def reset(self):
        self.plus_indicator = 0
        self.time = 0
        self.charged_visits = 0

        self.presampled_accumulations = self.random_state.binomial(self.horizon/self.acc_points.shape[0],
                                                                   self.p_c,
                                                                   size=self.acc_points.shape[0])

        self.count_sampling_steps=0
        if self.total_episodes <= self.avg_window_size:
            opt_rate = 0
        else:
            opt_rate = np.mean(self.optimal_choices[-self.avg_window_size:])

        available_actions = np.arange(self.action_space.n)
        self.acc_pointer = 0

        return self._get_obs(self.plus_indicator, self.charged_visits, self.time), {"optimality_rate": opt_rate, "available_actions": available_actions, "regretful_choices": self.regretful_choices}
    

    def _get_obs(self, plus_indicator, charged_visits, time):
        return np.array([plus_indicator, charged_visits, time])

    def step(self, action):
        reward = 0
        available_actions = np.arange(1)
        done = False

        if self.time == 0:
            self.plus_indicator = action
            self.c_0 = -1 + 2 * self.plus_indicator

            self.optimal_choices[self.total_episodes % self.avg_window_size] = float(action == self.get_optimal_actions())
            self.total_episodes += 1
            self.regretful_choices += float(action != self.get_optimal_actions())

        elif self.time < self.horizon:            

            if self.time in self.acc_points:
                self.charged_visits += self.presampled_accumulations[self.acc_pointer]
                self.count_sampling_steps += self.horizon/self.acc_points.shape[0]
                self.acc_pointer += 1

        else:
            done = True

            reward_bonus = self.bonus * self.plus_indicator
            reward_charged_visits = self.charged_visits * self.baseline_charged * self.c_0
            reward_s0 = -1 * self.c_0 * self.baseline_charged * self.horizon * self.p_c

            reward = reward_bonus + reward_charged_visits + reward_s0

            assert self.count_sampling_steps == self.horizon

        self.time += 1

        next_obs = self._get_obs(self.plus_indicator, self.charged_visits, self.time)

        return next_obs, reward, done, done, {"available_actions": available_actions}
    
    def get_optimal_actions(self):
        return  int(self.bonus > 0)
    
    def _get_first_action_values(self):
        return np.array([0, self.bonus])
    

class ChainAndSplitEnv(gym.Env):
    def __init__(self, num_actions=10, chain_length=10,
                 deterministic_bonus=0.05, max_stochastic_reward=1,
                 suboptimal_reward_categories=101, avg_window_size=750,
                 seed=0):
        super().__init__()

        self.num_actions = num_actions
        self.chain_length = chain_length
        self.suboptimal_reward_categories = suboptimal_reward_categories
        self.deterministic_bonus = deterministic_bonus
        self.max_stochastic_reward = max_stochastic_reward

        self.action_space = gym.spaces.Discrete(self.num_actions) # 0 = left, 1 = right
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, self.chain_length, self.suboptimal_reward_categories]),
                                                shape=(3,))

        self.right = None
        self.curr_step = None
        self.reward_cat = None
        self.reward_spacing = np.linspace(-self.max_stochastic_reward, self.max_stochastic_reward, self.suboptimal_reward_categories)
        
        self.random_state = np.random.RandomState(seed)

        self.avg_window_size = avg_window_size
        self.optimal_choices = np.zeros(self.avg_window_size)
        self.total_episodes = 0

    def reset(self):
        self.right = 0
        self.curr_step = 0
        self.reward_cat = 0

        if self.total_episodes <= self.avg_window_size:
            opt_rate = 0
        else:
            opt_rate = np.mean(self.optimal_choices[-self.avg_window_size:])

        action_set = np.arange(self.num_actions)

        return self._get_obs(), {"optimality_rate": opt_rate, "available_actions": action_set}
    
    def _get_obs(self):
        return np.array([self.right, self.curr_step, self.reward_cat], dtype=np.int32)
    
    def _get_first_action_values(self):
        q = np.zeros(self.num_actions)
        q[0] = self.deterministic_bonus
        return q
    
    def step(self, action):
        done = False
        reward = 0

        if self.curr_step == 0:
            if action == 0:
                self.right = 1
                self.optimal_choices[self.total_episodes % self.avg_window_size] = 1
            else:
                self.right = 0
                self.optimal_choices[self.total_episodes % self.avg_window_size] = 0

        self.curr_step += 1

        if (self.curr_step == 2) and (self.right == 0):
            self.reward_cat = self.random_state.randint(self.suboptimal_reward_categories)
            reward = self.reward_spacing[self.reward_cat]

        if (self.curr_step == self.chain_length):
            done = True
            if self.right == 1:
                reward = self.deterministic_bonus

            self.total_episodes += 1

        next_obs = self._get_obs()

        return next_obs, reward, done, done, {"available_actions": np.arange(1)}


class SimpleKeytoDoorWithDistractors(gym.Env):
    def __init__(self, num_steps=10, num_distractions=4, seed=0,
                 reduce_reward_to_scalar=False):
        super().__init__()

        self.num_actions = 2
        self.num_steps = num_steps
        self.num_distractions = num_distractions

        self.action_space = gym.spaces.Discrete(self.num_actions) # 0 = pick up key, 1 = open door

        # has key, door, distractions, curr_step, treasure
        self.observation_space = gym.spaces.Box(low=np.array([0] * (self.num_distractions + 4)), 
                                                high=np.array([1, 1] + [1] * self.num_distractions + [self.num_steps, 1]),
                                                shape=(self.num_distractions + 4,))
        
        self.distractor_weight = np.ones(self.num_distractions)
        self.distractor_weight = self.distractor_weight / (self.num_distractions * self.num_steps)

        self.reward_weights = np.zeros(self.num_distractions + 4)
        self.reward_weights[-1] = 1/self.num_steps
        self.reward_weights[2:-2] = self.distractor_weight
        
        self.random_state = np.random.RandomState(seed)

        self.has_key = None
        self.curr_step = None
        self.door = None
        self.distractions = None
        self.treasure = None

        self.regretful_choices = 0
        self.total_episodes = 0

        self.reduce_reward_to_scalar = reduce_reward_to_scalar

    def reset(self):
        self.has_key = 0
        self.curr_step = 0
        self.door = 0
        self.distractions = np.zeros(self.num_distractions, dtype=np.int32)
        self.treasure = 0

        return self._get_obs(), {"regretful_choices": self.regretful_choices}
    
    def _get_obs(self):
        return np.concatenate([[self.has_key, self.door], self.distractions, [self.curr_step, self.treasure]])
    
    def _get_reward(self, obs):
        reward_vec = obs * self.reward_weights
        if self.reduce_reward_to_scalar:
            return np.sum(reward_vec)
        return reward_vec
    
    def step(self, action):
        done = False

        if self.curr_step == 0:
            if action == 0:
                self.has_key = 1
            else:
                self.has_key = 0

        self.curr_step += 1
        self.distractions = self.random_state.binomial(1, 0.5, size=self.num_distractions)

        if self.curr_step == self.num_steps-1:
            self.door = 1
        else:
            self.door = 0

        if self.curr_step == self.num_steps:
            if (self.has_key == 1) and (action == 1):
                self.treasure = 1
                self.distractions = np.zeros(self.num_distractions, dtype=np.int32)
            else:
                self.treasure = 0
                self.distractions = np.zeros(self.num_distractions, dtype=np.int32)
                self.regretful_choices += 1

            done = True

        next_obs = self._get_obs()
        reward = self._get_reward(next_obs)

        if done:
            self.total_episodes += 1
        
        return next_obs, reward, done, done, {}