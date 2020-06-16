# import sys
# sys.path.append('.')
import numpy as np
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from utils import make_multiagent_env


class a2cTrainer():

    def __init__(self, n_agents, action_spaces, observation_spaces, writer, args):
        
        pass

    def eval(self):
        pass

    def prep_train(self):
        pass

    def sample_and_train(self):
        pass

    def reset(self):
        pass






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
    parser.add_argument("--centralized_actor", default=false, type=bool, help="Use centralized actor for action selection")
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
    timesteps = 0
    episode_rewards = [0.0]
    for ep in range(args.n_eps):
        observations = env.reset()
        trainer.reset()
        done = False
        for t in range(args.T):
            timesteps += 1
            actions = trainer.get_actions(observations)
            actions = [a.cpu().numpy() for a in actions]
            # print(actions)
            next_obs, rewards, dones, _ = env.step(actions)
            # trainer.store_transitions(*map_to_tensors(observations, actions, rewards, next_obs, dones))
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

