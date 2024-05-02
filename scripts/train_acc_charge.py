import numpy as np
import torch
import wandb

from algorithms.tabular_algo import TabularQFunctionBinaryStates
from environments.rl_envs import AccumulatedChargeChoice

from chunkers import CategoricalDeltaDynamicsModelChunker
from model import CategoricalStateDeltaDynamicsModel

from curr_utils import preprocess_actions, basic_preprocess_obs

from replay_buffer import ReplayBuffer

project_name = 'acc-charge-exps'

config_defaults = dict(
    algo='chunked-sarsa', # Algorithm to use (sarsa-lambda | chunked-sarsa)
    lr_model=0.0001,
    weight_decay_model=1e-6,
    lambda_td=0.8, # applicable for sarsa-lambda, not for chunked-sarsa
    #--------------------------
    replay_buffer_size=100000,
    train_every_x_frames=4,
    batch_size=128,
    init_fill_buffer=1000,
    #--------------------------
    # For the choice environment
    discount_factor=1.0,
    avg_window_size=3000,
    max_episodes=10000,
    horizon=200,
    accumulate_points=10,
    p_charged=0.5,
    bonus=0.1,
    seed=44,
    #--------------------------
    init_episodes_random=1000,
    lr=0.1/16,
    alg_epsilon=0.1,
    log_interval=1,
    #--------------------------
    log_episode_lambdas_every=500,
    #--------------------------
)


def main():
    wandb.init(config=config_defaults, project=project_name)
    args = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    env = AccumulatedChargeChoice(horizon=args.horizon,
                                    p_c=args.p_charged,
                                    acc_points=args.accumulate_points,
                                    bonus=args.bonus,
                                    seed=args.seed + 1,
                                    avg_window_size=args.avg_window_size)

    print("env-acc-points", env.acc_points)
    print("env-p-charged", env.p_c)
    
    if args.algo == 'sarsa-lambda':
        algo = TabularQFunctionBinaryStates(env=env, 
                                            gamma=args.discount_factor, 
                                            lr=args.lr, 
                                            epsilon=args.alg_epsilon, 
                                            update_rule=args.algo, 
                                            seed=args.seed)
        
    elif args.algo == 'chunked-sarsa':
            
        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        next_state_model = CategoricalStateDeltaDynamicsModel(env.observation_space,
                                                                env.action_space,
                                                                max_categories=args.horizon//args.accumulate_points + 1)
        
        chunker = CategoricalDeltaDynamicsModelChunker(env=env,
                                                        dynamics_model=next_state_model,
                                                        lr_dynamics_model=args.lr_model,
                                                        weight_decay_dynamics_model=args.weight_decay_model,
                                                        replay_buffer=replay_buffer,
                                                        preprocess_obs=basic_preprocess_obs,
                                                        preprocess_actions=preprocess_actions,
                                                        device=device,
                                                        seed=args.seed + 3)

        algo = TabularQFunctionBinaryStates(env=env,
                                            gamma=args.discount_factor, 
                                            lr=args.lr, 
                                            epsilon=args.alg_epsilon, 
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

    for episode in range(args.max_episodes):
        obs, info = env.reset()
        obs_list.append(obs)

        if episode < args.init_episodes_random:
            action = algo._act_randomly(obs, info['available_actions'])
            policy_at_obs = np.zeros(env.action_space.n)
            policy_at_obs[info['available_actions']] = 1/len(info['available_actions'])
        else:
            action = algo.act(obs, info['available_actions'])
            # get policy at obs
            policy_at_obs = algo._get_eps_greedy_policy_in_state(obs, info['available_actions'])

        done = False

        while not done:

            next_obs, reward, done, _, info_step = env.step(action)

            next_action = algo.act(next_obs, info_step['available_actions'])
            # get policy at next_obs
            policy_at_next_obs = algo._get_eps_greedy_policy_in_state(next_obs, info_step['available_actions']) 

            if args.algo == 'chunked-sarsa':
                # train model

                chunker.add_transition_to_replay_buffer(obs, action, next_obs, reward, done)

                if frames > args.batch_size and frames % args.train_every_x_frames == 0:
                    model_logs = chunker.train_model(args.batch_size)


                td_l = chunker.get_next_lambda(obs, action, next_obs, next_action, 
                                                policy_at_obs, policy_at_next_obs, done)

                td_l_list.append(td_l)

                algo.update_online(obs, action, reward, next_obs, done, next_action=next_action, td_lambda=td_l)
            else:
                algo.update_online(obs, action, reward, next_obs, done, next_action=next_action, td_lambda=args.lambda_td)


            obs_list.append(next_obs)
            policy_list.append(policy_at_obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            obs = next_obs
            action = next_action
            policy_at_obs = policy_at_next_obs
            frames += 1

        if episode % args.log_interval == 0:
            print(f'Episode: {episode}, Optimality rate: {info["optimality_rate"]}, Regretful choices: {info["regretful_choices"]}')
            wandb.log({'optimality_rate': info["optimality_rate"], 'epsilon': algo.epsilon , 'lr': algo.lr, 'frames': frames,
                        'episode': episode})

            if ('chunked' in args.algo) and frames > args.batch_size:
                print(f'Model loss: {model_logs["model_loss"]}')
                wandb.log({'model_loss': model_logs["model_loss"], 'frames': frames,
                           'episode': episode})



            wandb.log({'regretful_choices': info['regretful_choices'],
                       'episode': episode, 'frames': frames})

            if args.algo == 'chunked-sarsa':
                wandb.log({'td_lambda_mean': np.mean(td_l_list),
                            'td_lambda_std': np.std(td_l_list),
                            'td_lambda_max': np.max(td_l_list),
                            'td_lambda_min': np.min(td_l_list),
                            'episode': episode, 'frames': frames})

            td_l_list = []

        obs_list = []
        policy_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        episode += 1

    # final log    
    wandb.log({'regretful_choices': info['regretful_choices'], 'episode': episode, 'frames': frames})
    print("Final no. of regretful choices: ", info['regretful_choices'])


if __name__ == '__main__':
    main()
