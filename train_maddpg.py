# import sys
# sys.path.append('.')
import time
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
    parser.add_argument("--render_freq", default=0, type=int, help="Render frequency, if=0 no rendering")
    parser.add_argument("--use_writer", action="store_true", help="Use writer or no")
    parser.add_argument("--use_ounoise", action="store_true", help="Use OUNoise")
    parser.add_argument("--lograte", default=500, type=int, help="Log frequency")
    parser.add_argument("--train_freq", default=100, type=int, help="Training frequency")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--exp_name", default="maddpg test run", type=str, help="Experiment name")
    parser.add_argument("--algo", default="maddpg", type=str, help="What algo to use, maddpg or DDPG")
    parser.add_argument("--all_obs", action="store_true", help="Use all observations for every agent")
    parser.add_argument("--single_q", action="store_true", help="Use single Q-value network")
    parser.add_argument("--use_sac", action="store_true", help="Use soft actor-critic")
    parser.add_argument("--sac_alpha", default=0.01, type=float, help="What alpha hyperparameter to use")
    parser.add_argument("--use_td3", action="store_true", help="use  TD3 algorithm")
    parser.add_argument("--update_after", default=1, type=int, help="Update after this number of steps")

    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()

def prepro_observations(observations, all_obs=False):
    if all_obs:
        all_arr = np.concatenate(observations)
        return [np.copy(all_arr) for i in range(len(observations))]
    return observations


def learn_episodic_MADDPG(args):
    ###
    args.env = "simple_speaker_listener"
    # args.discrete_action = True
    env = make_multiagent_env(args.env)
    print(args.exp_name)
    print(args.sac_alpha)
    # print(act_sp)
    if not args.use_writer:
        print("not using writer")
    n_agents = len(env.agents)
    action_spaces = [act_sp.n for act_sp in env.action_space]
    observation_spaces = [ob_sp.shape[0] for ob_sp in env.observation_space]
    if args.all_obs:
        observation_spaces = [sum(observation_spaces) for i in range(n_agents)]
    # __import__('ipdb').set_trace()
    log_dir = "./logs/" + args.exp_name
    writer = SummaryWriter(log_dir) if args.use_writer else None
    running_rewards = deque([], maxlen=args.lograte)
    # discrete actions maddpg agentgent
    # agent = None
    trainer = MADDPG_Trainer(n_agents, action_spaces, observation_spaces, writer, args)
    trainer.eval()
    timesteps = 0
    episode_rewards = [0.0]
    for ep in range(args.n_eps):
        # print("EP" + str(ep))
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
            next_obs_ = prepro_observations(next_obs, args.all_obs)
            trainer.store_transitions(*map_to_tensors(observations, actions, rewards, next_obs_, dones))
            done = all(dones) or t >= args.T
            if timesteps > args.update_after and timesteps % args.train_freq == 0:

                trainer.prep_training()
                if args.use_sac:
                    # print("TRAINING SAC")
                    trainer.sample_and_train_sac(args.batch_size)
                elif not args.use_td3:
                    trainer.sample_and_train(args.batch_size)
                elif args.use_td3:
                    trainer.sample_and_train_td3(args.batch_size, args.train_freq)
                trainer.eval()
            observations = next_obs
            if args.render_freq != 0 and ep % args.render_freq == 0:
                # __import__('ipdb').set_trace()
                env.render()
            episode_rewards[-1] += np.sum(rewards)

            if done:
                break

        running_reward = episode_rewards[-1] / n_agents
        if args.use_writer:
            writer.add_scalar('ep_reward', running_reward, ep)
        running_rewards.append(running_reward)
        episode_rewards.append(0)
        if (ep + 1) % args.lograte == 0:
            print(f"episode: {ep}, running episode rewards: {np.mean(running_rewards)}")
        # TODO ADD logging to the
    # if args.use_writer: trainer.plot_actions()
    # eval_eps = 200
    # for i in range(eval_eps):
    #     observations = env.reset()
    #     trainer.reset()
    #     done = False
    #     for t in range(50):
    #         timesteps += 1
    #         actions = trainer.get_actions(observations)
    #         actions = [a.cpu().numpy() for a in actions]
    #         next_obs, rewards, dones, _ = env.step(actions)
    #         done = all(dones) or t >= args.T
    #         observations = next_obs
    #         time.sleep(0.2)
    #         env.render()
    #         if done:
    #             break
    # if args.use_writer:
    #     writer.export_scalars_to_json(str(log_dir / 'summary.json'))
    #     writer.close()
        
    return episode_rewards


if __name__ == '__main__':
    N_EPS = 10000
    args = get_args()
    rds_DDPG = learn_episodic_MADDPG(args)
    # print(args)
    # for i in range(10):
    #     rewards_DDPG = learn_episodic_MADDPG(args)
    #     rewards_numpy = np.asarray(rewards_DDPG)
    #     np.savetxt("ep_rewards/TD3-longer_discrete"+str(i) + ".csv", rewards_numpy, delimiter=',')
    # plt.legend()
    # plt.show()
