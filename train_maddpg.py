# import sys
# sys.path.append('.')
import numpy as np
import torch
import argparse
from collections import deque
from MADDPG_trainer import MADDPG_Trainer
# from torch.utils.tensorboard import SummaryWriter
from utils import make_multiagent_env, map_to_tensors


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="simple", type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=10000, type=int, help='N_episodes')
    parser.add_argument('--T', default=25, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_writer", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_ounoise", action="store_true", help="Use OUNoise")
    parser.add_argument("--lograte", default=100, type=int, help="Log frequency")
    parser.add_argument("--train_freq", default=100, type=int, help="Training frequency")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument('--save_dir', default='runs/checkpoints', help="directory where to save models")
    parser.add_argument("--load_file", default="", required=False, help="load model from specified file")
    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()
    

def learn_episodic_MADDPG(args):
    ###
    args.env = "simple_speaker_listener"
    # args.discrete_action = True
    env = make_multiagent_env(args.env)

    # print(act_sp)
    if not args.use_writer:
        print("not using writer")
    n_agents = len(env.agents)
    action_spaces = [act_sp.n for act_sp in env.action_space]
    observation_spaces = [ob_sp.shape[0] for ob_sp in env.observation_space]
    log_dir = "maddpg_test_run"
    writer = SummaryWriter(log_dir) if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    # discrete actions maddpg agentgent
    # agent = None
    trainer = MADDPG_Trainer(n_agents, action_spaces, observation_spaces, writer, args)
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
        observations = env.reset()
        trainer.reset()
        done = False
        for t in range(args.T):
            timesteps += 1
            actions = trainer.get_actions(observations)
            actions = [a.cpu().numpy() for a in actions]
            # print(actions)
            next_obs, rewards, dones, _ = env.step(actions)
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
    args = get_args()
    # rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
    rewards_DDPG = learn_episodic_MADDPG(args)
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legend()
    # plt.show()
