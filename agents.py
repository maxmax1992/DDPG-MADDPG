
import torch
import torch.optim as optim
import numpy as np
from NNets import MLPNetwork
from utils import soft_update, hard_update, gumbel_softmax
import seaborn as sns

sns.set(color_codes=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


class TD3_agent:

    def __init__(self, act_sp, ob_sp, all_obs, all_acts, hidden_dim=64,
                 start_steps=10000, update_after=1000, update_every=50):
        self.lr = 1e-2
        self.target_noise = 0.2
        self.target_noise_clip = 0.3
        self.act_noise = 0.1
        self.act_sp = act_sp
        self.ob_sp = ob_sp
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        # self.start_steps = 1
        # self.update_after = 1
        # self.update_every = 2
        print(f"act_sp: {act_sp}, ob_sp: {ob_sp}, all_obs: {all_obs}, all_acts: {all_acts}")
        self.policy = MLPNetwork(ob_sp, act_sp,
                                constrain_out=True, discrete_action=False,
                                td3_policy=True, hidden_dim=hidden_dim).to(device)
        self.policy_targ = MLPNetwork(ob_sp, act_sp,
                                constrain_out=True, discrete_action=False,
                                td3_policy=True, hidden_dim=hidden_dim).to(device)
        self.q_nets_n = 2
        self.qnets = []
        self.qnet_targs = []
        self.q_optimizers = []
        for i in range(self.q_nets_n):
            qnet = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)
            qnet_targ = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)
            qnet.to(device)
            qnet_targ.to(device) 
            hard_update(qnet_targ, qnet)
            self.qnets.append(qnet)
            self.qnet_targs.append(qnet_targ)
            self.q_optimizers.append(optim.Adam(qnet.parameters(), lr=self.lr))

        self.policy.to(device)
        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_targ.to(device)
        self.p_targ_optimizer = optim.Adam(self.policy_targ.parameters(), lr=self.lr)

        self.action_count = 0
        self.use_warmup = True

    def random_action(self):
        action = torch.zeros((1, self.act_sp)).to(device)
        action[0, np.random.randint(0, self.act_sp)] = 1.0
        return action

    def select_action(self, state, temperature=None, is_tensor=False, is_target=False):
        self.action_count += 1
        if self.action_count < self.start_steps:
            # return random action:
            # self.action
            return self.random_action()
        # print("select action")
        st = state
        if not is_tensor:
            st = torch.from_numpy(state).view(1, -1).float().to(device)
        if is_target:
            action = self.policy_targ(st)
            # action = self.policy_targ(st)
        else:
            # __import__('ipdb').set_trace()
            # print("not target")
            action = self.policy(st)
            noise = (self.act_noise**0.5)*torch.randn(action.shape)
            # __import__('ipdb').set_trace()
            action += noise
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        # __import__('ipdb').set_trace()
        return action_with_noise

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()
        for i in range(self.q_nets_n):
            self.qnets[i].train()
            self.qnet_targs[i].train()

    def update_targets(self, TAU):
        for i in range(self.q_nets_n):
            soft_update(self.qnet_targs[i], self.qnets[i], TAU)


class SAC_agent:

    def __init__(self, act_sp, ob_sp, all_obs, all_acts, hidden_dim=64):
        self.act_sp = act_sp
        self.ob_sp = ob_sp

        self.policy = MLPNetwork(ob_sp, act_sp, sac_policy=True,
                                constrain_out=True, hidden_dim=hidden_dim).to(device)
        self.q_nets_n = 2
        self.qnets = []
        self.qnet_targs = []
        self.q_optimizers = []
        for i in range(self.q_nets_n):
            qnet = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)
            qnet_targ = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)
            qnet.to(device)
            qnet_targ.to(device) 
            hard_update(qnet_targ, qnet)
            self.qnets.append(qnet)
            self.qnet_targs.append(qnet_targ)
            self.q_optimizers.append(optim.Adam(qnet.parameters(), lr=LR))

        self.policy.to(device)
        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.action_count = 0
        self.use_warmup = True

    def select_action(self, state, temperature=None, is_tensor=False, is_target=False):
        if self.use_warmup and self.action_count < WARMUP_STEPS:
            self.action_count += 1
            # TODO after finished: add temperature to Gumbel sampling
            # TODO ADD varmup steps to sac
        st = state
        if not is_tensor:
            st = torch.from_numpy(state).view(1, -1).float().to(device)
        if is_target:
            action = self.policy_targ(st)
            # action = self.policy_targ(st)
        else:
            # __import__('ipdb').set_trace()
            action = self.policy(st)
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        return action_with_noise

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()
        for i in range(self.q_nets_n):
            self.qnets[i].train()
            self.qnet_targs[i].train()

    def update_targets(self, TAU):
        for i in range(self.q_nets_n):
            soft_update(self.qnet_targs[i], self.qnets[i], TAU)

class DDPG_agent:

    def __init__(self, act_sp, ob_sp, all_obs, all_acts, hidden_dim=64):
        self.act_sp = act_sp
        self.ob_sp = ob_sp
        # print(ob_sp)
        print(f"ob_sp: {ob_sp} act_sp: {act_sp}")
        self.policy = MLPNetwork(ob_sp, act_sp, constrain_out=True, hidden_dim=hidden_dim).to(device)
        self.policy_targ = MLPNetwork(ob_sp, act_sp, constrain_out=True, hidden_dim=hidden_dim).to(device)
        self.qnet = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)
        self.qnet_targ = MLPNetwork(all_obs + all_acts, 1, constrain_out=False, hidden_dim=hidden_dim).to(device)

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
        # __import__('ipdb').set_trace()
        st = state
        if not is_tensor:
            st = torch.from_numpy(state).view(1, -1).float().to(device)
        if is_target:
            action = self.policy_targ(st)
        else:
            action = self.policy(st)
        action_with_noise = gumbel_softmax(action, hard=True).detach()
        return action_with_noise

    def update_targets(self):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)

    def set_eval(self):
        self.policy.eval()
        self.policy_targ.eval()

    def set_train(self):
        self.policy.train()
        self.policy_targ.train()
        self.qnet.train()
        self.qnet_targ.train()

    def update_targets(self, TAU):
        soft_update(self.policy_targ, self.policy, TAU)
        soft_update(self.qnet_targ, self.qnet, TAU)
