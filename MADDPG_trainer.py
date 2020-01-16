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
LR = 1e-2
TAU = 1e-2
WARMUP_STEPS = 20000
E_GREEDY_STEPS = 30000
INITIAL_STD = 2.0
FINAL_STD = 0.1
BATCH_SIZE = 64

def log_transition(self, state, action, reward, next_state, done):
        print("State", state)
        print("action", action)
        print("reward", reward)
        print("next state", next_state)
        print("done", done)

class DDPG_agent:

    def __init__(self, act_sp, ob_sp, all_obs, all_acts):
        self.act_sp = act_sp
        self.ob_sp = ob_sp

        self.policy = Policy_net(ob_sp, act_sp)
        self.policy_targ = Policy_net(ob_sp, act_sp)
        self.qnet = Q_net(all_obs, all_acts)
        self.qnet_targ = Q_net(all_obs, all_acts)

        self.policy.to(device)
        self.qnet.to(device)
        self.policy_targ.to(device)
        self.qnet_targ.to(device)

        hard_update(self.policy_targ, self.policy)
        hard_update(self.qnet_targ, self.qnet)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=LR)

    def select_action(self, state, temperature=None, is_tensor=False, is_target=False):
        # TODO after finished: add temperature to Gumbel sampling
        st = state
        if not is_tensor:
            st = torch.from_numpy(state).view(1, -1).float()
        if is_target:  
            action = self.policy_targ(st)
        else:
            action = self.policy(st)
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        return action_with_noise

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)


class MADDPG_Trainer:

    def __init__(self, n_agents, act_spcs, ob_spcs, writer, args):
        self.args = args
        self.memory = ReplayBuffer(int(1e6) // 2, len(act_spcs))
        self.epsilon_scheduler = LinearSchedule(E_GREEDY_STEPS, FINAL_STD, INITIAL_STD,
                                                warmup_steps=WARMUP_STEPS)
        self.n_agents = n_agents
        self.act_spcs = act_spcs
        self.ob_spcs = ob_spcs
        self.agents = [DDPG_agent(self.act_spcs[i], self.ob_spcs[i], np.sum(self.ob_spcs), \
            np.sum(self.act_spcs)) for i in range(n_agents)]

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

            

    def sample_and_train(self, batch_size):
        for i, agent in enumerate(self.agents):
            # Q_next = Q(s1, s2, p(s1), p(s2))
            batch = self.memory.sample(min(batch_size, len(self.memory)))

            states_i, actions_i, rewards_i, next_states_i, dones_i = batch
            actions_i = [[action.float().to(device) for action in acts] for acts in actions_i]
            # states_all = torch.stack[torch.cat(states_i)]
            actions_ = [self.agents[i].select_action(torch.stack(states_i[i]).to(device),
                        is_tensor=True, is_target=True) for i in range(self.n_agents)]

            q_input_obs, q_input_acts = self.transform_states(next_states_i, len(rewards_i)), \
                self.transform_actions(actions_, len(rewards_i))
            target_q = self.agents[i].qnet_targ(q_input_obs, q_input_acts).detach()
            rewards = torch.tensor([reward[i] for reward in rewards_i]).view(-1, 1).float().to(device)
            dones = torch.tensor([1 - done[i].float() for done in dones_i]).view(-1, 1).float().to(device)
            target_q = rewards + dones * GAMMA * target_q
            # states_all_vf = torch.stack([torch.cat(g_state) for g_state in states])
            input_acts = torch.stack([torch.cat(agents_actions) for agents_actions in actions_i]).float().to(device)
            input_obs = self.transform_states(states_i, len(rewards_i))
            
            input_q = self.agents[i].qnet(input_obs, input_acts)
            # print(input_q)
            self.agents[i].q_optimizer.zero_grad()
            loss = self.criterion(input_q, target_q)
            # print("LOSS", loss)
            loss.backward()
            self.agents[i].q_optimizer.step()
            
            # ACTOR gradient ascent of Q(s, π(s | ø)) with respect to ø
            if self.args.discrete_action:
                # use gumbel softmax max temp trick
                gumbel_sample = gumbel_softmax(self.agents[i].policy(torch.stack(states_i[i])), hard=True)
                # acts_to_forward = torch.stack([])
                q_obs_t = self.transform_states(states_i, len(rewards_i))

                for j, actions_batch in enumerate(actions_i):
                    actions_batch[i] = gumbel_sample[j]
                
                actions_t = torch.stack([torch.cat(agents_acts) for agents_acts in actions_i]).float()
                actor_loss = - self.agents[i].qnet(q_obs_t, actions_t).mean()
            else:
                # TODO handle continuous action space
                pass

            self.agents[i].p_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
            self.agents[i].p_optimizer.step()
            soft_update(self.agents[i].policy_targ, self.agents[i].policy, TAU)
            soft_update(self.agents[i].qnet_targ, self.agents[i].qnet, TAU)
            # if self.args.use_writer:
            #     self.writer.add_scalar("critic_loss", loss_critic.item(), self.n_updates)
            #     self.writer.add_scalar("actor_loss", actor_loss.item(), self.n_updates)
        
        #  CRITIC LOSS: Q(s, a) += (r + gamma*Q'(s, π'(s)) - Q(s, a))
        # onehot actions
        self.n_updates += 1
