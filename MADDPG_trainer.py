import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym.spaces import Box
from NNets import MLPNetwork
from utils import soft_update, hard_update, LinearSchedule, gumbel_softmax, onehot_from_logits, OUNoise
from buffer import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MEMORY_SIZE = int(1e6)
GAMMA = 0.99
LR = 1e-3
TAU = 1e-3
WARMUP_STEPS = 50000
E_GREEDY_STEPS = 100000
INITIAL_STD = 1.5
FINAL_STD = 0.01
BATCH_SIZE = 64

class DDPG_agent:

    def __init__(self, act_sp, ob_sp, continuous_action, hidden_dim=64, act_boundaries=(0, 1)):
        self.ob_sp = ob_sp
        self.continuous_action = continuous_action
        if self.continuous_action:
            self.exploration = OUNoise(act_sp)
        self.act_boundaries = act_boundaries
        self.policy = MLPNetwork(ob_sp, act_sp, constrain_out=True, hidden_dim=hidden_dim).to(device)
        self.policy_targ = MLPNetwork(ob_sp, act_sp, constrain_out=True, hidden_dim=hidden_dim).to(device)

        self.policy.to(device)
        self.policy_targ.to(device)

        hard_update(self.policy_targ, self.policy)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def select_greedy_from_target(self, states, use_target=True):
        policy = self.policy_targ if use_target else self.policy
        if self.continuous_action:
            return policy(states)
        else:
            return onehot_from_logits(policy(states))

    def select_action(self, state, temperature=None, is_tensor=False, is_target=False):
        # TODO after finished: add temperature to Gumbel sampling
        st = state
        if not is_tensor:
            st = torch.from_numpy(state).view(1, -1).float().to(device)
        if is_target:
            action = self.policy_targ(st)
        else:
            action = self.policy(st)
        if self.continuous_action:
            # print("CONT ACTION")
            action_with_noise = action.detach() + torch.from_numpy(self.exploration.noise()).float().to(device)
            action_with_noise = action_with_noise.clamp(*self.act_boundaries).detach()
            # print(self.act_boundaries)
            # print("ACTION: ", action_with_noise)
            return action_with_noise
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        # print("DISCRETE ACTION")
        return action_with_noise

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)

class MADDPG_Trainer:
    def __init__(self, n_agents, act_spcs, ob_spcs, writer, args):
        self.args = args
        self.memory = ReplayBuffer(args.buffer_length, n_agents, device)
        self.epsilon_scheduler = LinearSchedule(E_GREEDY_STEPS, FINAL_STD, INITIAL_STD,
                                                warmup_steps=WARMUP_STEPS)
        self.n_agents = n_agents
        self.ob_spcs = ob_spcs
        self.act_spcs = [1 if isinstance(act_spcs[i], Box) else act_spcs[i].n for i in range(n_agents)]
        self.qnet = MLPNetwork(np.sum(self.ob_spcs), np.sum(self.act_spcs), 1, constrain_out=False).to(device)
        self.qnet_targ = MLPNetwork(np.sum(self.ob_spcs), np.sum(self.act_spcs), 1, constrain_out=False).to(device)

        self.qnet.to(device)
        self.qnet_targ.to(device)

        hard_update(self.qnet_targ, self.qnet)

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=LR)
        self.agents = [DDPG_agent(self.act_spcs[i], self.ob_spcs[i],
                       isinstance(act_spcs[i], Box)) for i in range(n_agents)]
        self.n_steps = 0
        self.n_updates = 0
        self.writer = writer
        self.criterion = nn.MSELoss()

    def get_actions(self, states):
        return [agent.select_action(state)[0] for agent, state in zip(self.agents, states)]

    def store_transitions(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

    def reset(self):
        pass

    def transform_states(self, states, N):
        obses = []
        for i in range(N):
            states_ = []
            for j in range(self.n_agents):
                states_.append(states[j][i])
            obses.append(torch.cat([f.float().to(device) for f in states_]))
        return torch.stack(obses)

    def transform_actions(self, actions, N):
        acts = []
        for i in range(N):
            actions_ = []
            for j in range(self.n_agents):
                actions_.append(actions[j][i])
            acts.append(torch.cat([f.float().to(device) for f in actions_]))
        return torch.stack(acts)

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.policy_targ, agent.policy, TAU)
            soft_update(agent.qnet_targ, agent.qnet, TAU)

    def prep_training(self):
        for agent in self.agents:
            agent.qnet.train()
            agent.policy.train()
            agent.qnet_targ.train()
            agent.policy_targ.train()

    def get_save_data(self):
        # TODO
        pass

    def load_models(self, checkpoints):
        # TODO
        pass

    def eval(self):
        for agent in self.agents:
            agent.qnet.eval()
            agent.policy.eval()
            agent.qnet_targ.eval()
            agent.policy_targ.eval()

    def sample_and_train(self, batch_size):
        # TODO ADD Model saving, optimize code
        batch = self.memory.sample(min(batch_size, len(self.memory)))

        states_i, actions_i, rewards_i, next_states_i, dones_i = batch

        states_all = torch.cat(states_i, 1)
        next_states_all = torch.cat(next_states_i, 1)
        actions_all = torch.cat(actions_i, 1)
        for i, agent in enumerate(self.agents):
            next_actions_all = [ag.select_greedy_from_target(next_state)
                                for ag, next_state in zip(self.agents, next_states_i)]
            # computing target
            total_obs = torch.cat([next_states_all, torch.cat(next_actions_all, 1)], 1)
            target_q = self.agents[i].qnet_targ(total_obs).detach()
            rewards = rewards_i[i].view(-1, 1)
            dones = dones_i[i].view(-1, 1)
            target_q = rewards + (1 - dones) * GAMMA * target_q

            # computing the inputs
            input_q = self.agents[i].qnet(torch.cat([states_all, actions_all], 1))
            self.agents[i].q_optimizer.zero_grad()
            loss = self.criterion(input_q, target_q.detach())
            # print("LOSS", loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].qnet.parameters(), 0.5)
            self.agents[i].q_optimizer.step()
            actor_loss = 0
            # ACTOR gradient ascent of Q(s, π(s | ø)) with respect to ø
            # use gumbel softmax max temp trick
            policy_out = self.agents[i].policy(states_i[i])
            if not self.agents[i].continuous_action:
                acts_agent = gumbel_softmax(policy_out, hard=True)
            else:
                acts_agent = policy_out

            actions_curr_pols = [agent_.select_greedy_from_target(state, use_target=False)
                                 for agent_, state in zip(self.agents, states_i)]

            for action_batch in actions_curr_pols:
                action_batch.detach_()
            actions_curr_pols[i] = acts_agent

            actor_loss = - self.agents[i].qnet(torch.cat([states_all.detach(),
                                               torch.cat(actions_curr_pols, 1)], 1)).mean()
            if not self.agents[i].continuous_action:
                actor_loss += (policy_out**2).mean() * 1e-3

            self.agents[i].p_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.agents[i].policy.parameters(), 0.5)
            self.agents[i].p_optimizer.step()
            # detach the forward propagated action samples
            actions_i[i].detach_()

            if self.args.use_writer:
                self.writer.add_scalars("Agent_%i" % i, {
                    "vf_loss": loss,
                    "actor_loss": actor_loss
                }, self.n_updates)
        self.update_all_targets()
        self.n_updates += 1
