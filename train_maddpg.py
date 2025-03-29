# import sys
# sys.path.append('.')
import gc
import time
import numpy as np
import argparse
import torch
from collections import deque
from MADDPG_trainer import MADDPG_Trainer
from torch.utils.tensorboard import SummaryWriter
from utils import make_multiagent_env, map_to_tensors
from mem_report import mem_report
from config import default_config
# from pytorch_memlab import MemReporter

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default=default_config.env, type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=default_config.n_eps, type=int, help='N_episodes')
    parser.add_argument('--T', default=default_config.T, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render_freq", default=default_config.render_freq, type=int, help="Render frequency, if=0 no rendering")
    parser.add_argument("--use_writer", action="store_true", help="Use writer or no", default=default_config.use_writer)
    parser.add_argument("--use_ounoise", action="store_true", help="Use OUNoise", default=default_config.use_ounoise)
    parser.add_argument("--lograte", default=default_config.lograte, type=int, help="Log frequency")
    parser.add_argument("--train_freq", default=default_config.train_freq, type=int, help="Training frequency")
    parser.add_argument("--buffer_length", default=default_config.buffer_length, type=int)
    parser.add_argument("--batch_size", default=default_config.batch_size, type=int, help="Batch size for training")
    parser.add_argument("--exp_name", default=default_config.exp_name, type=str, help="Experiment name")
    parser.add_argument("--algo", default=default_config.algo, type=str, help="What algo to use, maddpg or DDPG")
    parser.add_argument("--all_obs", action="store_true", help="Use all observations for every agent", default=default_config.all_obs)
    parser.add_argument("--single_q", action="store_true", help="Use single Q-value network", default=default_config.single_q)
    parser.add_argument("--use_sac", action="store_true", help="Use soft actor-critic", default=default_config.use_sac)
    parser.add_argument("--sac_alpha", default=default_config.sac_alpha, type=float, help="What alpha hyperparameter to use")
    parser.add_argument("--use_td3", action="store_true", help="use  TD3 algorithm", default=default_config.use_td3)
    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()

def prepro_observations(observations, all_obs=False):
    if all_obs:
        all_arr = np.concatenate(observations)
        return [np.copy(all_arr) for i in range(len(observations))]
    return observations


def learn_episodic_MADDPG(args):
    ###
    # Initialize the environment based on args
    print(f"Creating environment: {args.env}")
    env = make_multiagent_env(args.env)
    print(f"Experiment: {args.exp_name}")
    
    # Get number of agents and spaces
    n_agents = len(env.agents)
    action_spaces = [act_sp.n for act_sp in env.action_space]
    observation_spaces = [ob_sp.shape[0] for ob_sp in env.observation_space]
    
    if args.all_obs:
        observation_spaces = [sum(observation_spaces) for i in range(n_agents)]
    
    # Set up logging
    log_dir = "./logs/" + args.exp_name
    writer = SummaryWriter(log_dir) if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    
    # Initialize trainer
    trainer = MADDPG_Trainer(n_agents, action_spaces, observation_spaces, writer, args)
    trainer.eval()
    
    timesteps = 0
    episode_rewards = [0.0]
    # memreporter = MemReporter()
    # trainer.eval()
    for ep in range(args.n_eps):
        observations = env.reset()
        trainer.reset()
        done = False
        for t in range(args.T):
            timesteps += 1
            # preprocess observations
            observations = prepro_observations(observations, args.all_obs)
            
            # get actions
            actions = trainer.get_actions(observations)
            actions = [a.cpu().numpy() for a in actions]
            # print(actions)
            next_obs, rewards, dones, _ = env.step(actions)
            # __import__('ipdb').set_trace()
            # rewards = [rew**2 for rew in rewards]
            # with torch.no_grad():
            # next_obs_ = prepro_observations(next_obs, args.all_obs)
            trainer.store_transitions(observations, actions, rewards, next_obs, dones)
            done = all(dones) or t >= args.T
            if timesteps % args.train_freq == 0:
                trainer.prep_training()
                if args.use_sac:
                    # print("TRAINING SAC")
                    trainer.sample_and_train_sac(args.batch_size)
                elif not args.use_td3:
                    trainer.sample_and_train(args.batch_size)
                elif args.use_td3:
                    trainer.train_td3(args.batch_size)
                # gc.collect()
                trainer.eval()
            observations = next_obs
            # if args.use_td3 and ep > 1:
            #     trainer.sample_and_train_td3(args.batch_size)
            if args.render_freq != 0 and ep % args.render_freq == 0:
                # __import__('ipdb').set_trace()
                env.render()
            episode_rewards[-1] += np.sum(rewards)

            if done:
                break

        running_reward = episode_rewards[-1] / n_agents
        # print(running_reward)
        if args.use_writer:
            writer.add_scalar('ep_reward', running_reward, ep)
        running_rewards.append(running_reward)
        episode_rewards.append(0)
        if (ep + 1) % args.lograte == 0:
            # memreporter.report()
            # print(len(trainer.memory))
            gc.collect()
            print(f"episode: {ep}, running episode rewards: {np.mean(running_rewards)}")
        # TODO ADD logging to the
    # if args.use_writer: trainer.plot_actions()
    eval_eps = 5  # Reduced from 1000 to make it faster
    for i in range(eval_eps):
        observations = env.reset()
        trainer.reset()
        done = False
        for t in range(40):
            timesteps += 1
            actions = trainer.get_actions(observations)
            actions = [a.cpu().numpy() for a in actions]
            next_obs, rewards, dones, _ = env.step(actions)
            done = all(dones) or t >= args.T
            observations = next_obs
            time.sleep(0.07)
            env.render()
            if done:
                break
    # if args.use_writer:
    #     writer.export_scalars_to_json(str(log_dir / 'summary.json'))
    #     writer.close()
        
    return episode_rewards 

def main():
    # Parse arguments or use defaults from config.py
    args = get_args()
    
    # Run training with the specified arguments
    rewards_MADDPG = learn_episodic_MADDPG(args)
    
    # Save rewards to file
    rewards_numpy = np.asarray(rewards_MADDPG)
    np.savetxt(f"ep_rewards/MADDPG_{args.env}.csv", rewards_numpy, delimiter=',')
    
    return 0

if __name__ == '__main__':
    main()
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legend()
    # plt.show()
