import torch
import wandb
import numpy as np

from algorithms.decomposed_tabular_algo import DecomposedTabularAlgo
from environments.rl_envs import SimpleKeytoDoorWithDistractors

from chunkers import CategoricalDynamicsModelChunker
from model import CategoricalStateDynamicsModel

from curr_utils import basic_preprocess_obs

from replay_buffer import ReplayBuffer

project_name = 'keytodoor'

config_defaults = dict(
    # Algorithm ---------------------------
    algo='chunked-expected-sarsa', # (expected-sarsa-lambda, chunked-expected-sarsa)
    reduce_lambda_to_scalar=False, # False with chunked-expected-sarsa is C-factored in the paper. True corresponds to C-default.
    seed=123,
    lr=0.1/4,
    lambda_td=0.9, # applicable for expected-sarsa-lambda, not the chunked variants
    discount_factor=1.0,
    # Exploration -------------------------
    init_epsilon=1,
    final_epsilon=0.1,
    decay_over_fraction_episodes=0.1, # fraction of max_episodes over which epsilon is decayed from init_epsilon to final_epsilon
    # Environment -------------------------
    num_steps=100,
    num_distractions=4,
    max_episodes=5000,
    # Model and chunker -------------------
    use_mask=False, # masking for each separate classification problem in the chunked model
    action_embedding_dim=4,
    replay_buffer_size=10000,
    model_lr=0.0002,
    train_model_every=1,
    batch_size=64,
    # Device ------------------------------
    use_cpu=False,
    # Logging -----------------------------
    log_interval=10,
)

def main():
    wandb.init(config=config_defaults, project=project_name)
    args = wandb.config

    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    
    env = SimpleKeytoDoorWithDistractors(num_steps=args.num_steps, 
                                            num_distractions=args.num_distractions,
                                            seed=args.seed,
                                            reduce_reward_to_scalar=False,)
        
    algo = DecomposedTabularAlgo(env.observation_space, 
                                 env.action_space,
                                 update_rule=args.algo,
                                 gamma=args.discount_factor, 
                                 lr=args.lr, 
                                 epsilon=args.init_epsilon,
                                 seed=args.seed,
                                 )
    
    if args.algo == 'chunked-expected-sarsa':
        replay_buffer = ReplayBuffer(args.replay_buffer_size)

        dynamics_model = CategoricalStateDynamicsModel(env.observation_space,
                                                    env.action_space,
                                                    action_embedding_dim=args.action_embedding_dim)

        chunker = CategoricalDynamicsModelChunker(env,
                                                dynamics_model,
                                                lr_dynamics_model=args.model_lr,
                                                replay_buffer=replay_buffer,
                                                preprocess_obs=basic_preprocess_obs,
                                                device=device,
                                                seed=args.seed,
                                                use_mask=args.use_mask,)

    epsilon_decay = (args.init_epsilon - args.final_epsilon) / (args.max_episodes * args.decay_over_fraction_episodes)

    for episode in range(1, args.max_episodes+1):
        obs, info = env.reset()
        done = False

        algo.set_epsilon(max(args.final_epsilon, algo.epsilon - epsilon_decay))

        action = algo.act(obs)
        policy_at_obs = algo._get_eps_greedy_policy_at_state(obs)
        tstep=0

        while not done:
            tstep+=1
            next_obs, reward, d1, d2, _ = env.step(action)
            done = d1 or d2
            next_action = algo.act(next_obs)
            policy_at_next_obs = algo._get_eps_greedy_policy_at_state(next_obs)



            if args.algo in ['chunked-expected-sarsa']:
                # add to replay buffer
                chunker.add_transition_to_replay_buffer(obs, action, next_obs, reward, done)

                # train model
                if (chunker.replay_buffer.__len__() >= args.batch_size) and (tstep % args.train_model_every == 0):
                    model_logs = chunker.train_model(batch_size=args.batch_size)


            if args.algo == 'expected-sarsa-lambda':
                algo.update_online(state=obs,
                                    action=action,
                                    reward_vector=reward,
                                    next_state=next_obs,
                                    done=done,
                                    policy_in_next_state=policy_at_next_obs,
                                    lambda_td=np.array(args.lambda_td))
            elif args.algo == 'chunked-expected-sarsa':

                l_td_vector = chunker.get_next_lambda(obs, action, next_obs, next_action, 
                                                      policy_at_obs, policy_at_next_obs, done,
                                                      reduce_to_scalar=args.reduce_lambda_to_scalar)
                

                algo.update_online(state=obs,
                                    action=action,
                                    reward_vector=reward,
                                    next_state=next_obs,
                                    done=done,
                                    policy_in_next_state=policy_at_next_obs,
                                    lambda_td=l_td_vector)
            else:
                raise NotImplementedError

            obs = next_obs
            action = next_action

        if episode % args.log_interval == 0:
            print('Episode {}\tRegretful Choices: {:.2f}'.format(
                episode, info['regretful_choices']))
            print("Q-table for components x actions at initial state: ",  algo.q_tables[:, 0, 0, 0, 0, 0, 0, 0, 0])

            if args.algo in ['chunked-expected-sarsa']:
                wandb.log({'episode': episode,
                           'epsilon': algo.epsilon,
                           'regretful_choices': info['regretful_choices'],
                           'model_loss': model_logs['model_loss'],})
            else:
                wandb.log({'episode': episode,
                           'epsilon': algo.epsilon,
                           'regretful_choices': info['regretful_choices'],})
            
    # final log
    obs, info = env.reset()

    print('Episode {}\tRegretful Choices: {:.2f}'.format(
        episode, info['regretful_choices']))
    
    wandb.log({'episode': episode,
                'final_regretful_choices': info['regretful_choices'],})


if __name__ == '__main__':
    main()

