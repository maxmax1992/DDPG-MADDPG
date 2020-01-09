# import sys
# sys.path.append('.')
import numpy as np
import argparse
from collections import deque
from MADDPG_trainer import MADDPG_Trainer
from torch.utils.tensorboard import SummaryWriter
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
    parser.add_argument("--discrete_action", default=False, type=bool, help="Use discrete action outputs")
    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()


def learn_episodic_DDPG(args):
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

    writer = SummaryWriter() if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    # discrete actions maddpg agentgent
    # agent = None
    trainer = MADDPG_Trainer(n_agents, action_spaces, observation_spaces, writer, args)
    for ep in range(args.n_eps):
        observations = env.reset()
        trainer.reset()
        done = False
        epr = 0
        for t in range(args.T):
            actions = trainer.get_actions(observations)
            actions = [a.cpu().numpy() for a in actions]
            print(actions)
            next_obs, rewards, dones, _ = env.step(actions)
            trainer.store_transitions(*map_to_tensors(observations, actions, rewards, next_obs, dones))
            done = all(dones) or t >= args.T
            trainer.train_agents()
            epr += np.mean(rewards)
            observations = next_obs

            if args.render:
                env.render()

            if done:
                break

        if args.use_writer:
            writer.add_scalar('Epr', epr, ep)
        running_rewards.append(epr)
        if (ep + 1) % args.lograte == 0:
            print(f"episode: {ep}, running episode rewards: {np.mean(running_rewards)}")
        # TODO ADD logging to the
        
    return 0


if __name__ == '__main__':
    N_EPS = 10000
    args = get_args()
    # rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
    rewards_DDPG = learn_episodic_DDPG(args)
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legend()
    # plt.show()