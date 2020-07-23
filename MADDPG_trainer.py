import torch
import torch.nn as nn
import numpy as np
from utils import gumbel_softmax, onehot_from_logits
from buffer import ReplayBuffer
import seaborn as sns
import matplotlib.pyplot as plt
from agents import SAC_agent, DDPG_agent, TD3_agent
sns.set(color_codes=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
MEMORY_SIZE = int(1e6)
GAMMA = 0.95
LR = 1e-2
TAU = 1e-2
WARMUP_STEPS = 10000
E_GREEDY_STEPS = 30000
# INITIAL_STD = 2.0
# FINAL_STD = 0.1
BATCH_SIZE = 64
TD3_random_act_prob = 0.05



class MADDPG_Trainer:

    def __init__(self, n_agents, act_spcs, ob_spcs, writer, args):
        self.args = args
        self.memory = ReplayBuffer(args.buffer_length, n_agents, device)
        # self.memory = ReplayMemory(args.buffer_length, n_agents, device)
        self.use_maddpg = args.algo == "maddpg"
        self.use_sac = args.use_sac
        self.use_td3 = args.use_td3
        self.use_single_q = args.single_q
        self.all_obs = args.all_obs
        self.n_agents = n_agents
        self.act_spcs = act_spcs
        self.ob_spcs = ob_spcs
        qnet_actspcs = [np.sum(self.act_spcs) if self.use_maddpg else self.act_spcs[i]
                        for i in range(n_agents)]
        qnet_obspcs = [np.sum(self.ob_spcs) if self.use_maddpg else self.ob_spcs[i]
                        for i in range(n_agents)]
        if self.use_sac and not self.use_td3:
            self.agents = [SAC_agent(self.act_spcs[i], qnet_obspcs[i] if self.all_obs
                                     else self.ob_spcs[i], qnet_obspcs[i],
                           qnet_actspcs[i]) for i in range(n_agents)]
        elif self.use_td3:
            self.agents = [TD3_agent(self.act_spcs[i], qnet_obspcs[i] if self.all_obs
                                      else self.ob_spcs[i], qnet_obspcs[i],
                           qnet_actspcs[i]) for i in range(n_agents)]
        else:
            self.agents = [DDPG_agent(self.act_spcs[i], qnet_obspcs[i] if self.all_obs
                                      else self.ob_spcs[i], qnet_obspcs[i],
                           qnet_actspcs[i]) for i in range(n_agents)]
        self.n_steps = 0
        self.n_updates = 0
        self.writer = writer
        self.criterion = nn.MSELoss()
        self.sac_alpha = args.sac_alpha
        self.agent_actions = [[] for i in range(self.n_agents)]

    def plot_actions(self):
        for i in range(self.n_agents):
            sns.distplot(self.agent_actions[i], bins=self.agents[i].act_sp, kde=False)
            # __import__('ipdb').set_trace()
            plt.show()

    def get_actions(self, states):
        result = []
        # with torch.no_grad():
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action = agent.select_action(state)[0]
            result.append(action)
            # if self.args.use_writer: self.agent_actions[i].append(np.argmax(action.cpu()).item())
        self.n_steps += 1
        return result


    def store_transitions(self, states, actions, rewards, next_states, dones):
        # print(sys.getsizeof(states) + sys.getsizeof(actions) + sys.getsizeof(rewards)
        #       + sys.getsizeof(next_states) + sys.getsizeof(dones))
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
            agent.update_targets(TAU)

    def prep_training(self):
        for agent in self.agents:
            agent.set_train() 

    def eval(self):
        for agent in self.agents:
            agent.set_eval()

    def sample_and_train_td3(self, batch_size):
        t = self.n_steps
        # print(self.n_steps) 
        update_every = self.agents[0].update_every
        update_after = self.agents[0].update_after
        if (t + 1) > update_after and (t + 1) % update_every == 0:
            for i in range(update_every):
                 self.train_td3(batch_size, i)

    def batch_add_random_acts(self, tensor, ag_i):
        # __import__('ipdb').set_trace()
        n_clip  =self.agents[ag_i].target_noise_clip
        noise = (self.agents[ag_i].target_noise**0.5)*torch.randn(tensor.shape)
        noise = torch.clamp(noise, -n_clip, n_clip)
        tensor[:] = tensor[:] + noise
        # __import__('ipdb').set_trace()

    def train_td3(self, batch_size):
        self.n_updates += 1
        batch = self.memory.sample(min(batch_size, len(self.memory)))
        states_i, actions_i, rewards_i, next_states_i, dones_i = batch
        # __import__('ipdb').set_trace()
        if self.use_maddpg:
            states_all = torch.cat(states_i, 1)
            next_states_all = torch.cat(next_states_i, 1)
            actions_all = torch.cat(actions_i, 1)
        for i, agent in enumerate(self.agents):
            # print("training_qnet")
            if not self.use_maddpg:
                states_all = states_i[i]
                next_states_all = next_states_i[i]
                actions_all = actions_i[i]
            if self.use_maddpg:  
                next_actions_all = [ag.policy(next_state)
                                    for ag, next_state in zip(self.agents, next_states_i)]

                [self.batch_add_random_acts(e, i) for i, e in enumerate(next_actions_all)]
                next_actions_all = [onehot_from_logits(e) for e in next_actions_all]
            else:
                actions_and_logits = [onehot_from_logits(agent.policy(next_states_i[i]))]
                next_actions_all = [e[0] for e in actions_and_logits]
            total_obs = torch.cat([next_states_all, torch.cat(next_actions_all, 1)], 1)
            qnet_targs = []
            for qnet in self.agents[i].qnet_targs:
                qnet_targs.append(qnet(total_obs).detach())
            rewards = rewards_i[i].view(-1, 1)
            dones = dones_i[i].view(-1, 1)
            qnet_mins = torch.min(qnet_targs[0], qnet_targs[1])
            target_q = rewards + (1 - dones) * GAMMA * (qnet_mins)
            losses = []
            for j, qnet in enumerate(self.agents[i].qnets):
                input_q = qnet(torch.cat([states_all, actions_all], 1))
                self.agents[i].q_optimizers[j].zero_grad()
                loss = self.criterion(input_q, target_q.detach())
                losses.append(loss.item())
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(qnet.parameters(), 0.5)
                self.agents[i].q_optimizers[j].step()


        if self.args.use_writer:
            self.writer.add_scalar(f"Agent_{i}: q_net_loss: ", np.mean(losses), self.n_updates)
        if self.n_updates % 2 == 0:
            for i in range(self.n_agents):
                # print("training policy")
                actor_loss = 0
                # ACTOR gradient ascent of Q(s, π(s | ø)) with respect to ø
                # use gumbel softmax max temp trick
                policy_out = self.agents[i].policy(states_i[i])
                gumbel_sample = gumbel_softmax(policy_out, hard=True)
                if self.use_maddpg:
                    actions_curr_pols = [onehot_from_logits(agent_.policy(state))
                                         for agent_, state in zip(self.agents, states_i)]

                    for action_batch in actions_curr_pols:
                        action_batch.detach_()
                    actions_curr_pols[i] = gumbel_sample
                    actor_loss = - self.agents[i].qnets[0](torch.cat([states_all.detach(),
                                                       torch.cat(actions_curr_pols, 1)], 1)).mean()
                else:
                    actor_loss = - self.agents[i].qnets[0](torch.cat([states_all.detach(),
                                                       gumbel_sample], 1)).mean()
                self.agents[i].p_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agents[i].policy.parameters(), 0.5)
                self.agents[i].p_optimizer.step()
                actions_i[i].detach_()
                if self.args.use_writer:
                    self.writer.add_scalar(f"Agent_{i}: policy_objective: ", actor_loss.item(), self.n_updates)
                self.update_all_targets()
        # self.n_updates += 1

    def sample_and_train_sac(self, batch_size):
        # TODO ADD Model saving, optimize code
        batch = self.memory.sample(min(batch_size, len(self.memory)))
        states_i, actions_i, rewards_i, next_states_i, dones_i = batch
        # __import__('ipdb').set_trace()        
        if self.use_maddpg:
            states_all = torch.cat(states_i, 1)
            next_states_all = torch.cat(next_states_i, 1)
            actions_all = torch.cat(actions_i, 1)
        for i, agent in enumerate(self.agents):
            if not self.use_maddpg:
                states_all = states_i[i]
                next_states_all = next_states_i[i]
                actions_all = actions_i[i]
            if self.use_maddpg:  
                actions_and_logits = [onehot_from_logits(ag.policy(next_state), logprobs=True)
                                    for ag, next_state in zip(self.agents, next_states_i)]

                next_actions_all = [e[0] for e in actions_and_logits]
                next_logits_all = [self.sac_alpha*e[1] for e in actions_and_logits]
                # __import__('ipdb').set_trace()
            else:
                actions_and_logits = [onehot_from_logits(agent.policy(next_states_i[i]),
                                                       logprobs=True)]
                next_actions_all = [e[0] for e in actions_and_logits]
                next_logits_all = [self.sac_alpha*e[1] for e in actions_and_logits]
                
            # computing target
            total_obs = torch.cat([next_states_all, torch.cat(next_actions_all, 1)], 1)
            
            # target_q = self.agents[i].qnet_targ(total_obs).detach()
            qnet_targs = []
            for qnet in self.agents[i].qnet_targs:
                qnet_targs.append(qnet(total_obs).detach())
            rewards = rewards_i[i].view(-1, 1)
            dones = dones_i[i].view(-1, 1)
            qnet_mins = torch.min(qnet_targs[0], qnet_targs[1])
            # __import__('ipdb').set_trace()
            logits_idx = i if self.use_maddpg else 0
            logits_agent = next_logits_all[logits_idx]
            # if len(qnet_mins.squeeze(-1)) != len(logits_agent.squeeze(-1)):
            #     __import__('ipdb').set_trace()
            target_q = rewards + (1 - dones) * GAMMA * (qnet_mins -
                                     logits_agent.reshape(qnet_mins.shape))
            # __import__('ipdb').set_trace()
            # computing the inputs
            for j, qnet in enumerate(self.agents[i].qnets):
                input_q = qnet(torch.cat([states_all, actions_all], 1))
                self.agents[i].q_optimizers[j].zero_grad()
                # print("----")
                # __import__('ipdb').set_trace() 
                loss = self.criterion(input_q, target_q.detach())
                # print('after')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 0.5)
                self.agents[i].q_optimizers[j].step()

            # __import__('ipdb').set_trace()
            actor_loss = 0
            # ACTOR gradient ascent of Q(s, π(s | ø)) with respect to ø
            # use gumbel softmax max temp trick
            policy_out = self.agents[i].policy(states_i[i])
            gumbel_sample, act_logprobs = gumbel_softmax(policy_out, hard=True, logprobs=True)
            act_logprobs = self.sac_alpha*act_logprobs
            # __import__('ipdb').set_trace() 
            if self.use_maddpg:
                with torch.no_grad():
                    actions_curr_pols = [onehot_from_logits(agent_.policy(state))
                                         for agent_, state in zip(self.agents, states_i)]
                actions_curr_pols[i] = gumbel_sample
                total_obs = torch.cat([states_all, torch.cat(actions_curr_pols, 1)], 1)
                qnet_outs = []
                for qnet in self.agents[i].qnets:
                    qnet_outs.append(qnet(total_obs))
                qnet_mins = torch.min(qnet_outs[0], qnet_outs[1])
                actor_loss = - qnet_mins.mean()
                # __import__('ipdb').set_trace()
            else:
                # actor_loss = - self.agents[i].qnet(torch.cat([states_all.detach(),
                #                                    gumbel_sample], 1)).mean()
                # actions_curr_pols[i] = gumbel_sample
                # __import__('ipdb').set_trace()
                total_obs = torch.cat([states_all, gumbel_sample], 1)
                qnet_outs = []
                for qnet in self.agents[i].qnets:
                    qnet_outs.append(qnet(total_obs))
                qnet_mins = torch.min(qnet_outs[0], qnet_outs[1])
                actor_loss = - qnet_mins.mean()
            # actor_loss += (policy_out**2).mean() * 1e-3

            self.agents[i].p_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
            # torch.nn.utils.clip_grad_norm_(self.agents[i].policy.parameters(), 0.5)
            self.agents[i].p_optimizer.step()
            # detach the forward propagated action samples
            actions_i[i].detach_()
            # __import__('ipdb').set_trace()
            if self.args.use_writer:
                self.writer.add_scalars("Agent_%i" % i, {
                    "vf_loss": loss,
                    "actor_loss": actor_loss
                }, self.n_updates)
        
        self.update_all_targets()
        self.n_updates += 1

    def sample_and_train(self, batch_size):
        return
        # TODO ADD Model saving, optimize code
        batch = self.memory.sample(min(batch_size, len(self.memory)))
        states_i, actions_i, rewards_i, next_states_i, dones_i = batch
        # __import__('ipdb').set_trace()        
        if self.use_maddpg:
            states_all = torch.cat(states_i, 1)
            next_states_all = torch.cat(next_states_i, 1)
            actions_all = torch.cat(actions_i, 1)
        for i, agent in enumerate(self.agents):
            if not self.use_maddpg:
                states_all = states_i[i]
                next_states_all = next_states_i[i]
                actions_all = actions_i[i]
            if self.use_maddpg:  
                next_actions_all = [onehot_from_logits(ag.policy_targ(next_state))
                                    for ag, next_state in zip(self.agents, next_states_i)]
            else:
                next_actions_all = [onehot_from_logits(agent.policy_targ(next_states_i[i]))]
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
            gumbel_sample = gumbel_softmax(policy_out, hard=True)
            if self.use_maddpg:
                actions_curr_pols = [onehot_from_logits(agent_.policy(state))
                                     for agent_, state in zip(self.agents, states_i)]

                for action_batch in actions_curr_pols:
                    action_batch.detach_()
                actions_curr_pols[i] = gumbel_sample

                actor_loss = - self.agents[i].qnet(torch.cat([states_all.detach(),
                                                   torch.cat(actions_curr_pols, 1)], 1)).mean()
            else:
                actor_loss = - self.agents[i].qnet(torch.cat([states_all.detach(),
                                                   gumbel_sample], 1)).mean()
            actor_loss += (policy_out**2).mean() * 1e-3

            self.agents[i].p_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.agents[i].policy.parameters(), 0.5)
            self.agents[i].p_optimizer.step()
            # detach the forward propagated action samples
            actions_i[i].detach_()
            # __import__('ipdb').set_trace()
            if self.args.use_writer:
                self.writer.add_scalars("Agent_%i" % i, {
                    "vf_loss": loss,
                    "actor_loss": actor_loss
                }, self.n_updates)
        self.update_all_targets()
        self.n_updates += 1
