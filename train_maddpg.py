# import sys
# sys.path.append('.')
import numpy as np
import argparse
from collections import deque
from MADDPG_trainer import MADDPG_Trainer
from torch.utils.tensorboard import SummaryWriter
from utils import make_multiagent_env, map_to_tensors
from env_wrappers import SubprocVecEnv, DummyVecEnv

def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_multiagent_env(env_id)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="simple", type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=25000, type=int, help='N_episodes')
    parser.add_argument('--T', default=25, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_writer", action="store_true", help="Render the environment mode")
    parser.add_argument("--use_ounoise", action="store_true", help="Use OUNoise")
    parser.add_argument("--lograte", default=100, type=int, help="Log frequency")
    parser.add_argument("--discrete_action", default=True, type=bool, help="Use discrete action outputs")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    return parser.parse_args()


def learn_episodic_DDPG(args):
    ###
    args.env = "simple_speaker_listener"
    # args.discrete_action = True
    env = make_parallel_env(args.env, args.n_rollout_threads, args.seed)

    # print(act_sp)
    if not args.use_writer:
        print("not using writer")
    n_agents = len(env.action_space)
    action_spaces = [act_sp.n for act_sp in env.action_space]
    observation_spaces = [ob_sp.shape[0] for ob_sp in env.observation_space]

    writer = SummaryWriter() if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    # discrete actions maddpg agentgent
    # agent = None
    trainer = MADDPG_Trainer(n_agents, action_spaces, observation_spaces, writer, args)
    timesteps = 0
    for ep in range(args.n_eps):
        observations = env.reset()
        trainer.reset()
        done = False
        epr = 0
        print("episode ", ep)
        for t in range(args.T):
            actions = trainer.get_actions(observations)
            actions = [[a.cpu().numpy() for a in actions]]
            # actions = [a.cpu().numpy() for a in actions]
            # print(actions)
            next_obs, rewards, dones, _ = env.step(actions)
            print("NEXT OBS", next_obs[0])
            trainer.store_transitions(*map_to_tensors(observations, actions, rewards, next_obs[0], dones))
            done = all(dones) or t >= args.T
            if timesteps % args.steps_per_update == 0:
                trainer.sample_and_train(args.batch_size)
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