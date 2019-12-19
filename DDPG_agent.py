import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from NNets import Q_net, Policy_net
from utils import soft_update, hard_update, ReplayMemory, LinearSchedule, \
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

class DDPG_Agent:

    def __init__(self, ob_sp, act_sp, alow, ahigh, writer, args):
        self.args = args
        self.alow = alow
        self.ahigh = ahigh
        self.act_sp = act_sp
        self.policy = Policy_net(ob_sp, act_sp)
        self.policy_targ = Policy_net(ob_sp, act_sp)
        self.qnet = Q_net(ob_sp, act_sp)
        self.qnet_targ = Q_net(ob_sp, act_sp)
    
        self.policy.to(device)
        self.qnet.to(device)
        self.policy_targ.to(device)
        self.qnet_targ.to(device)
        self.MSE_loss = nn.MSELoss()
        self.noise = OUNoise(1, 1)

        hard_update(self.policy_targ, self.policy)
        hard_update(self.qnet_targ, self.qnet)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.memory = ReplayMemory(int(1e6))
        self.epsilon_scheduler = LinearSchedule(E_GREEDY_STEPS, FINAL_STD, INITIAL_STD,
                                                warmup_steps=WARMUP_STEPS)
        self.n_steps = 0
        self.n_updates = 0
        self.writer = writer

    def get_action(self, state):
        if self.args.use_ounoise:
            noise = self.noise.sample()
        else:
            noise = np.random.normal(0, self.epsilon_scheduler.value(self.n_steps), self.act_sp)
        st = torch.from_numpy(state).view(1, -1).float()
        action = self.policy(st)
        if self.args.discrete_action:
            # action_with_noise = action + torch.from_numpy(noise).to(device).float()
            # action_with_noise = action_with_noise.argmax().item()
            action_with_noise = gumbel_softmax(action, hard=True).argmax().item()
        else:
            action_with_noise = np.clip(action.item() + noise[0], self.alow, self.ahigh)
        if self.args.use_writer:
            self.writer.add_scalar("action mean", action.item(), self.n_steps)
            self.writer.add_scalar("action noise", noise, self.n_steps)
            self.writer.add_scalar("epsilon", self.epsilon_scheduler.value(self.n_steps), self.n_steps)
            self.writer.add_scalar("action", action_with_noise, self.n_steps)
        self.n_steps += 1
        return action_with_noise

    def store_transition(self, state, action, reward, next_state, done):

        self.memory.push(torch.from_numpy(state), torch.tensor(action),
                         torch.tensor(reward), torch.from_numpy(next_state),
                         torch.tensor(done))

    def reset(self):
        self.noise.reset()

    def train(self):
        batch = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        b_dict = [torch.stack(elem) for elem in Transition(*zip(*batch))]
        states, actions, rewards, next_states, dones = \
            b_dict[0], b_dict[1].view(-1, 1), \
            b_dict[2].view(-1, 1).float().to(device), b_dict[3], \
            b_dict[4].view(-1, 1).float().to(device)

        #  CRITIC LOSS: Q(s, a) += (r + gamma*Q'(s, π'(s)) - Q(s, a))
        # onehot actions
        if self.args.discrete_action:
            actions_one_hot = torch.zeros(actions.shape[0], self.act_sp).float().to(device)
            actions_one_hot[torch.arange(actions.shape[0]), actions.view(-1)] = 1
            # forward of the policy
            inputs_critic = self.qnet(states, actions_one_hot)
        else:
            inputs_critic = self.qnet(states, actions)
        # targets
        with torch.no_grad():
            # get one-hot actions from
            policy_acts = onehot_from_logits(self.policy_targ(next_states))
        targ_values = self.qnet_targ(next_states, policy_acts)
        targets_critics = rewards + GAMMA * (1 - dones) * targ_values
        loss_critic = self.MSE_loss(inputs_critic, targets_critics)
        self.q_optimizer.zero_grad()
        loss_critic.backward()
        # nn.utils.clip_grad_norm_(self.qnet.parameters(), 5)
        self.q_optimizer.step()

        # ACTOR gradient ascent of Q(s, π(s | ø)) with respect to ø
        if self.args.discrete_action:
            # use gumbel softmax max temp trick
            gumbel_sample = gumbel_softmax(self.policy(states), hard=True)
            actor_loss = - self.qnet(states, gumbel_sample).mean()
        else:
            actor_loss = - self.qnet(states, self.policy(states)).mean()

        self.p_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.p_optimizer.step()
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)
        if self.args.use_writer:
            self.writer.add_scalar("critic_loss", loss_critic.item(), self.n_updates)
            self.writer.add_scalar("actor_loss", actor_loss.item(), self.n_updates)
        self.n_updates += 1
