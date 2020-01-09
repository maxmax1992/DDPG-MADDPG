import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from NNets import Q_net, Policy_net
from utils import soft_update, hard_update, ReplayBuffer, LinearSchedule, \
    Transition, OUNoise, gumbel_softmax, onehot_from_logits

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MEMORY_SIZE = int(1e6)
GAMMA = 0.99
LR = 1e-3
TAU = 1e-3
WARMUP_STEPS = 20000
E_GREEDY_STEPS = 30000
INITIAL_STD = 2.0
FINAL_STD = 0.1
BATCH_SIZE = 64


class DDPG_agent:

    def __init__(self, act_sp, ob_sp):
        self.act_sp = act_sp
        self.ob_sp = ob_sp

        self.policy = Policy_net(ob_sp, act_sp)
        self.policy_targ = Policy_net(ob_sp, act_sp)
        self.qnet = Q_net(ob_sp, act_sp)
        self.qnet_targ = Q_net(ob_sp, act_sp)

        self.policy.to(device)
        self.qnet.to(device)
        self.policy_targ.to(device)
        self.qnet_targ.to(device)

        hard_update(self.policy_targ, self.policy)
        hard_update(self.qnet_targ, self.qnet)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=LR)

    def select_action(self, state, temperature=None):
        # TODO add temperature to Gumbel sampling
        st = torch.from_numpy(state).view(1, -1).float()
        action = self.policy(st)
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        return action_with_noise

    def q_train(self, i, states, actions, rewards, next_states, dones):
        """
        Rewards for this agent. 1xN array
        States, array of states
        """
        
        # states =
        print(1)
        pass

    def p_train(self, i, states):
        
        pass

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)


class MADDPG_Trainer:

    def __init__(self, n_agents, act_spcs, ob_spcs, writer, args):
        self.args = args
        self.memory = ReplayBuffer(int(1e6) // 2)
        self.epsilon_scheduler = LinearSchedule(E_GREEDY_STEPS, FINAL_STD, INITIAL_STD,
                                                warmup_steps=WARMUP_STEPS)
        self.n_agents = n_agents
        self.act_spcs = act_spcs
        self.ob_spcs = ob_spcs
        self.agents = [DDPG_agent(self.act_spcs[i], self.ob_spcs[i]) for i in range(n_agents)]

        self.n_steps = 0
        self.n_updates = 0
        self.writer = writer

    def get_actions(self, states):
        return [agent.select_action(state)[0] for agent, state in zip(self.agents, states)]

    def store_transitions(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

    def reset(self):
        pass

    def train_agents(self):
        batch = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        states, actions, rewards, next_states, dones = batch
        states_all_vf = torch.stack([torch.cat(g_state) for g_state in states])
        
        vf_next_in = []
        # actions_stacked = actions.
        for i, agent in enumerate(self.agents):
            agent.q_train(i, states, actions, rewards, next_states, dones)
            agent.p_train(i, states)
            agent.update_targets()
        
        #  CRITIC LOSS: Q(s, a) += (r + gamma*Q'(s, π'(s)) - Q(s, a))
        # onehot actions
        self.n_updates += 1
