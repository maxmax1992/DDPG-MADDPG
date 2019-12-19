import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
from tensorboardX import SummaryWriter

def moving_average(x, N):
    return np.convolve(x, np.ones(N, ), mode='valid') / N

# taken from openAI baselines
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
N_EPS = 500

class Policy_net(nn.Module):
    def __init__(self):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(3, 200)
        self.affine2 = nn.Linear(200, 100)
        self.mean_head = nn.Linear(100, 1)
        # self.sigma = torch.nn.Parameter(torch.tensor([self.sigma], requires_grad=True))

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        # -2 to 2 with tanh
        mean = 2*torch.tanh(self.mean_head(x))
        return  mean

class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        action_space = 1
        self.affine1 = nn.Linear(3 + action_space, 200)
        self.affine2 = nn.Linear(200, 100)
        self.value_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        return self.value_head(x)
    
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Q_value_net = Q_net()
Q_value_net_target = Q_net()
hard_update(Q_value_net_target, Q_value_net)

policy_net = Policy_net()
policy_net_target = Policy_net()
hard_update(policy_net_target, policy_net)

Q_value_net_target.eval()
policy_net_target.eval()

optim_q = optim.Adam(Q_value_net.parameters(), lr=1e-3)
optim_p = optim.Adam(policy_net.parameters(), lr=1e-3)

# def select_greedy(obs):
#     with torch.no_grad():
#         obs_ = torch.from_numpy(obs).float()
#         values = Q_network(obs_)
#         return torch.argmax(values.detach()).view(1, -1)
    
# def get_action(observation):
#     x = torch.from_numpy(observation).float()
#     normal = policy_net(x)
#     return normal.sample()

def get_q_inputs(state, action):
    return torch.cat([state, action], 1)

def train_on_batch(memory, batch_size, df, T, writer):
    # TODO-in future: remove the casting to tensors all the time
    # Vectorized implementation
    batch = memory.sample(batch_size)
    # connect all batch Transitions to one tuple
    batch_n = Transition(*zip(*batch))
    # reshape actions so ve can collect the DQN(S_t, a_t) easily with gather
    actions = torch.tensor(batch_n.action).float().view(-1, 1)
    # get batch states
#     print(batch_n.state)
    states = torch.cat(batch_n.state).float()
    next_states = torch.cat(batch_n.next_state).float()
    batch_rewards = torch.cat(batch_n.reward).float().view(-1, 1)
    
    dones = torch.tensor(batch_n.done).float().view(-1, 1)
    # collect only needed Q-values with corresponding actions for loss computation
    inputs = Q_value_net(get_q_inputs(states, actions))
    targets = batch_rewards
    # targets += df*Q_network_target(next_states).max(1)[0].detach()*(1 - dones)
    targets += (1-dones)*df*Q_value_net_target(torch.cat([next_states, policy_net_target(next_states).view(-1, 1)], 1))
    
    # critic loss
    optim_q.zero_grad()
    loss = F.mse_loss(inputs, targets.view(inputs.shape))
    writer.add_scalar("Critic loss", loss.item(), T)
    loss.backward()
    optim_q.step()

    # actor loss
    optim_p.zero_grad()
    loss_actor = -1 * (Q_value_net(torch.cat([states, policy_net(states).view(-1, 1)], 1))).mean()
    writer.add_scalar("Actor loss", loss_actor.item(), T)
    loss_actor.backward()
    optim_q.step()

    soft_update(Q_value_net_target, Q_value_net, tau=0.001)
    soft_update(policy_net_target, policy_net, tau=0.001)
    
def learn_episodic_DDPG(N_eps=500): 
    
    memory_len = 1000000
    df = 0.99
    batch_size = 64
    train_freq = 16
    T = 0
    # target_update_freq = 1000
    warmup_steps = 10000
    # scheduler
    e_s = 1.5
    e_e = 0.10
    N_decay = 60000
    scheduler = LinearSchedule(N_decay, e_e, e_s)
    
    # replay mem
    memory = ReplayMemory(memory_len)
    rewards = []
    writer = SummaryWriter()

    env = gym.make('Pendulum-v0')
    # n_actions = env.action_space.n
    actions = []
    for i_episode in range(N_eps):
        
        observation = env.reset()
        total_r = 0

        for t in range(200):
            T += 1
            if T < warmup_steps:
                action = env.action_space.sample()[0]
                # print(action)
            else:
                curr_epsilon = scheduler.value(T - warmup_steps)
                noise = np.random.normal(0, curr_epsilon)
                action_mean = policy_net(torch.from_numpy(observation).float())
                action = np.clip(action_mean.item() + noise , -2.0, 2.0)
            writer.add_scalar("action", action, T)
            # print(action)
            next_observation, reward, done, info = env.step([action])
            total_r += reward
            reward = torch.tensor([reward])
            
            memory.push(torch.from_numpy(observation).view(1, -1), \
                action, reward, torch.from_numpy(next_observation).view(1, -1), float(done))
            
            # train the DQN
            if T % train_freq == 0:
                train_on_batch(memory, min(batch_size, T), df, T, writer)
            
            observation = next_observation
                
            if done:
                writer.add_scalar("Episode_reward", total_r, i_episode)
                if (i_episode + 1) % 100 == 0:
                    # print('curr eps', curr_epsilon)
                    print("Episode {} finished with {} total rewards, T: {}".format(i_episode, total_r, T))
                break
                
        rewards.append(total_r)

    # render environment
    for i in range(5):
        observation = env.reset()
        for j in range(500):
            action_mean = policy_net(torch.from_numpy(observation).float())
            action = np.clip(action_mean.item(), -2.0, 2.0)
            next_observation, reward, done, info = env.step([action])
            env.render()
    env.close()
    
    return rewards

# rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
rewards_DDPG = learn_episodic_DDPG(N_EPS*2)

plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
plt.legend()
plt.show()