# import sys
# sys.path.append('.')
import gym
import numpy as np
import torch
import argparse
from collections import deque
from MADDPG_trainer import MADDPG_Trainer
from torch.utils.tensorboard import SummaryWriter
from utils import make_multiagent_env, map_to_tensors
from new_envs.mnist_trainer import Mnist_hyperparam_env
from gym.spaces import Box

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="simple", type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=10000, type=int, help='N_episodes')
    parser.add_argument('--T', default=100000, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_writer", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_ounoise", action="store_true", help="Use OUNoise")
    parser.add_argument("--lograte", default=100, type=int, help="Log frequency")
    parser.add_argument("--train_freq", default=100, type=int, help="Training frequency")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument('--save_dir', default='runs/checkpoints', help="directory where to save models")
    parser.add_argument("--load_file", default="", required=False, help="load model from specified file")
    parser.add_argument("--hyperopt_env", action="store_true", help="Use hyperOptim env")
    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()

def reset_env(env, nagents):
    obs = env.reset()
    return [np.copy(obs) for _ in range(nagents)]

def learn_episodic_MADDPG(args, env=None):

    env = gym.make("BipedalWalker-v2")
    # print(act_spk)
    if not args.use_writer:
        print("not using writer")
    n_agents = 4
    env.action_space = [Box(-1, 1, [1]) for _ in range(n_agents)]
    # action_spaces = [act_sp.n for act_sp in env.action_space]
    observation_spaces = [24 for _ in range(n_agents)]
    log_dir = "bipedal_logging"
    writer = SummaryWriter(log_dir) if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    # discrete actions maddpg agentgent
    # agent = None
    trainer = MADDPG_Trainer(n_agents, env.action_space, observation_spaces, writer, args)
    # import ipdb; ipdb.set_trace()
    trainer.eval()
    # TODO maybe change this to specified amount
    model_save_rate = 100
    episode_rewards = [0.0]
    timesteps = 0
    start_ep = 0
    # try to load from givern checkpoint file
    if args.load_file != "":
        try:
            # load dict file
            checkpoint = torch.load(args.load_file)
            trainer.load_models(checkpoint)
            timesteps = checkpoint['timesteps']
            start_ep = checkpoint['start_ep']
        except Exception:
            print("Unable to load from specified checkpoint")

    for ep in range(start_ep, args.n_eps):
        observations = reset_env(env, n_agents)
        trainer.reset()
        done = False
        for t in range(args.T):
            timesteps += 1
            actions = trainer.get_actions(observations)
            # TODO fix action to better
            actions = [a.detach().cpu().numpy() for a in actions]
            # print(actions)
            next_ob, reward, done, _ = env.step(actions)
            next_obs = [np.copy(next_ob) for _ in range(n_agents)]
            rewards = [np.float(reward) for _ in range(n_agents)]
            dones = [done for _ in range(n_agents)]
            trainer.store_transitions(*map_to_tensors(observations, actions, rewards, next_obs, dones))
            done = all(dones) or t >= args.T
            if timesteps % args.train_freq == 0:
                trainer.prep_training()
                trainer.sample_and_train(args.batch_size)
                trainer.eval()
            observations = next_obs

            if args.render:
                env.render()

            episode_rewards[-1] += np.sum(rewards)

            if done:
                break
            
        # if (ep + 1) % model_save_rate == 0:
        #     savedir = args.save_dir + 'episode' + str(ep)
        #     torch.save({
        #         'timesteps': timesteps,
        #         'start_ep': ep,
        #         **trainer.get_save_data()
        #     }, savedir)

        if args.use_writer:
            writer.add_scalar('rewards', episode_rewards[-1] / n_agents, ep)
        running_rewards.append(episode_rewards[-1] / n_agents)
        episode_rewards.append(0)
        if (ep + 1) % args.lograte == 0:
            print(f"episode: {ep}, running episode rewards: {np.mean(running_rewards)}")
        # TODO ADD logging to the
    writer.export_scalars_to_json(str(log_dir / 'summary.json'))
    writer.close()
        
    return 0

if __name__ == '__main__':
    N_EPS = 10000
    env = None
    args = get_args()
    print("ARGS", args)
    # rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
    if args.hyperopt_env:
        env = Mnist_hyperparam_env()
    
    rewards_DDPG = learn_episodic_MADDPG(args, env)
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legen d()
    # plt.show()
