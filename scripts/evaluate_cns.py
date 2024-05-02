import numpy as np
import wandb

from algorithms.tabular_algo import TabularQFunctionBinaryStates
from environments.rl_envs import ChainAndSplitEnv
from chunkers import TrulyTabularModelChunker

from model import TrulyTabularModel

project_name = 'chunktd-cns'

config_defaults = dict(
    algo='chunked-sarsa', # Algorithm to use (sarsa-lambda | chunked-sarsa)
    use_model='truly-tabular',
    lr=0.1/256, #/32 for sarsa(0) and /2048 for sarsa(1),
    lambda_td=0.0, # applicable for sarsa-lambda, not for chunked-sarsa
    #--------------------------
    # For the environment
    discount_factor=1.0,
    max_episodes=100001,
    seed=45,
    chain_length=20,
    num_actions=10,
    deterministic_bonus=0.01,
    max_stochastic_reward=1.0,
    suboptimal_reward_categories=101,
    #--------------------------
    log_interval=1,
)


def main():
    wandb.init(config=config_defaults, project=project_name)
    args = wandb.config
    
    env = ChainAndSplitEnv(num_actions=args.num_actions,
                        chain_length=args.chain_length,
                        deterministic_bonus=args.deterministic_bonus,
                        max_stochastic_reward=args.max_stochastic_reward,
                        suboptimal_reward_categories=args.suboptimal_reward_categories,
                        seed=args.seed + 1)
    
    if args.algo == 'sarsa-lambda':
        algo = TabularQFunctionBinaryStates(env=env, 
                                            gamma=args.discount_factor, 
                                            lr=args.lr,  
                                            update_rule=args.algo, 
                                            seed=args.seed)
        
    elif args.algo == 'chunked-sarsa':

        if args.use_model == 'truly-tabular':
            next_state_model = TrulyTabularModel(env.observation_space, env.action_space)
            chunker = TrulyTabularModelChunker(dynamics_model=next_state_model)

        algo = TabularQFunctionBinaryStates(env=env,
                                            gamma=args.discount_factor, 
                                            lr=args.lr,
                                            update_rule=args.algo, 
                                            seed=args.seed,
                                            )
    else:
        raise ValueError('Unknown algorithm')
    
    print("q-table shape", algo.q_table.shape)
    

    obs_list = []
    policy_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    td_l_list = []
    frames = 0
    episode = 0

    delta_with_opt_list = []

    for episode in range(args.max_episodes):
        obs, info = env.reset()
        obs_list.append(obs)

        # act randomly
        action = algo._act_randomly(obs, info['available_actions'])
        policy_at_obs = np.zeros(env.action_space.n)
        policy_at_obs[info['available_actions']] = 1/len(info['available_actions'])

        done = False

        while not done:

            next_obs, reward, done, _, info_step = env.step(action)

            # act randomly
            next_action = algo._act_randomly(next_obs, info_step['available_actions'])
            policy_at_next_obs = np.zeros(env.action_space.n)
            policy_at_next_obs[info_step['available_actions']] = 1/len(info_step['available_actions'])


            if args.algo == 'sarsa-lambda':
                algo.update_online(obs, action, reward, next_obs, done, next_action=next_action, td_lambda=args.lambda_td)
            else:

                # train (tabular) model on transition
                chunker.train_model([obs], [action], [next_obs])

                # get lambda
                td_l = chunker.get_next_lambda(obs, action, next_obs, next_action, 
                                                policy_at_obs, policy_at_next_obs, done)

                td_l_list.append(td_l)

                algo.update_online(obs, action, reward, next_obs, done, next_action=next_action, td_lambda=td_l)
                    


            obs_list.append(next_obs)
            policy_list.append(policy_at_obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            obs = next_obs
            action = next_action
            policy_at_obs = policy_at_next_obs
            frames += 1


        delta_with_opt_list.append(algo.q_table[0, 0, 0][0] - np.max(algo.q_table[0, 0, 0][1:]))

        if episode % args.log_interval == 0:
            print(f'Episode: {episode}, q-delta-with-opt: {delta_with_opt_list[-1]}')
                
            wandb.log({'q_delta_with_opt': delta_with_opt_list[-1],
                        'epsilon': algo.epsilon , 'lr': algo.lr,
                        'episode': episode, 'frames': frames})
                    
            delta_with_opt_list = []
            td_l_list = []

        obs_list = []
        policy_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        episode += 1


if __name__ == '__main__':
    main()
