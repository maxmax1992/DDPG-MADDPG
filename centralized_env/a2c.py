import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import gym
from gym import spaces
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import A2C, SAC, PPO2
from tensorboardX import SummaryWriter

from utils.make_env import make_env
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv_


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    # if n_rollout_threads == 1:
    return DummyVecEnv_([get_env_fn(0)])
    # else:
    #     return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def make_env_2(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env

# creates centralized env from multienv
class Centralized_multienv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, scenario, arglist):
        super(Centralized_multienv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # simple_speaker_listener 1 1 False
        self.env = make_env_2(scenario, [])
        # self.env = make_env_2(scenario, [])
        
        # env2 = make_env
        self.actions_list = []
        first_asp, second_asp = self.env.action_space[0].n, self.env.action_space[1].n
        self.action_spaces = [space.n for space in self.env.action_space]
        for i in range(first_asp):
            for j in range(second_asp):
                self.actions_list.append((i, j))
        print(self.actions_list)
        self.N_DISCRETE_ACTIONS = first_asp * second_asp
        # self.N_DISCRETE_ACTIONS = self.action_spaces[0] + 

        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        # todo change this to another non hardcoded value
        self.observation_space = spaces.Box(
            low=-1e38, high=1e38, shape=(14,), dtype=np.float32)
        # self.observation_space = self.env.observation_space[0]
        self.i = 0
        self.ep_rewards = [0.0]
        # print("HERE")
        self.writer = SummaryWriter(arglist.logdir)
        self.ep_steps = 0

    def oneHotEncode(self, action, nactions):
        actions_vector = np.zeros(nactions)
        actions_vector[action] = 1
        return actions_vector

    def step(self, action):
        # if len(self.ep_rewards) > 6000 and len(self.ep_rewards) < 6025:
        #     self.render()
        # TODO change this to use argument list
        terminal = (self.ep_steps >= 25)
        action_tuple = list(self.actions_list[action])
        action_tuple = [self.oneHotEncode(
            action_tuple[i], self.action_spaces[i]) for i in range(len(action_tuple))]
        # action = oneHotEncode(action, self.action_spaces[0])
        # print(action)
        # __import__('ipdb').set_trace()
        obs_n, reward_n, done_n, info_n = self.env.step(action_tuple)
        next_state = np.concatenate(obs_n)
        reward_total = np.sum(reward_n)
        done = all(done_n) or terminal
        self.ep_rewards[-1] += reward_total
        if done:
            self.ep_steps = 0
            self.writer.add_scalar('data/sum_of_rewards', self.ep_rewards[-1]/2, len(self.ep_rewards))
            self.ep_rewards.append(0)
        self.ep_steps += 1
        return next_state[0], reward_total, done, info_n
        # return this obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = self.env.reset()
        # print("111")
        state = np.concatenate(obs_n)
        # print('2222')
        # __import__('ipdb').set_trace() 
        return state

    def render(self, mode='human', close=False):
        self.env.render()


def oneHotEncode(action, nactions):
    actions_vector = np.zeros(nactions)
    actions_vector[action] = 1
    return actions_vector


def callback(locals_, globals_):
    pass

# def action_dim_test():
#     env = make_env('simple_speaker_listener', [])
#     state0 = env.reset()
#     a1 = oneHotEncode(1, 3)
#     a2 = oneHotEncode(3, 5)
#     state1, _, _, _ = env.step([a1, a2])
#     assert len(state0[0]) + len(state0[1]
#                                 ) == len(state1[0]) + len(state1[1]), "qqq"
#     a1 = oneHotEncode(1, 3)
#     a2 = oneHotEncode(3, 5)
#     state3, _, _, _ = env.step([a1, a2])
#     assert len(state0[0]) + len(state0[1]
#                                 ) == len(state3[0]) + len(state3[1]), "qqq"
#     a1 = oneHotEncode(1, 3)
#     a2 = oneHotEncode(3, 5)
#     state2, _, _, _ = env.step([a1, a2])
#     assert len(state0[0]) + len(state0[1]
#                                 ) == len(state2[0]) + len(state2[1]), "qqq"


if __name__ == '__main__':
    # action_dim_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str,
                        default="runs/Centralized_PPO_debug_env")
    arglist = parser.parse_args()
    n_episodes = 40000
    steps_per_ep = 25

    # env = Centralized_multienv('simple', arglist)
    env = Centralized_multienv('simple_speaker_listener', arglist)
    env = DummyVecEnv([lambda: env])
    # Define and Train the agent
    # :param n_env: (int) The number of environments to run
    # :param n_steps: (int) The number of steps to run for each environment
    # :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    # model = A2C(MlpPolicy, env, tensorboard_log=arglist.logdir)
    # p_kwargs = {"is_discrete": True}#, "n_env": 1, "n_steps": 1, "n_batch": None}
    # model = SAC(MlpPolicy, env, policy_kwargs=p_kwargs)
    model = PPO2(MlpPolicy, env)
    env.reset()
    
    model.learn(total_timesteps=n_episodes*steps_per_ep, callback=callback) 

# TODOLIST
# 0. Clean code and push thing to the git.
# 1. Verify reward logging in maddpgpytorch,
# 2. Try to figure out why the centralized actor does not work well ????
