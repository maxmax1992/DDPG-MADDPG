import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, 
                 constrain_out=False, sac_policy=False, norm_in=True, 
                 discrete_action=True, td3_policy=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        self.td3_policy = td3_policy
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.sac_policy = sac_policy
        if self.sac_policy:
            self.fc3_var = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            # print('AAAAA')
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
            # __import__('ipdb').set_trace()
        else:  # logits for discrete action (will softmax later)
            
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        # __import__('ipdb').set_trace()
        out = self.out_fn(self.fc3(h2))
        if self.sac_policy:
            var_outputs = self.fc3_var(h2)
            noise = torch.normal(torch.zeros_like(var_outputs),
                                 torch.ones_like(var_outputs)).detach()
            var_outputs *= noise
            out = out + var_outputs

        if self.td3_policy:
            # print("HERE")
            return out*2.0

        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Policy_net(nn.Module):

    def __init__(self, ob_sp, act_sp):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(ob_sp, 60)
        self.affine2 = nn.Linear(60, 60)
        self.mean_head = nn.Linear(60, act_sp)

    def forward(self, x):
        x = x.to(device).float()
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        mean = 2 * torch.tanh(self.mean_head(x))
        return mean


class Q_net(nn.Module):
    def __init__(self, ob_sp, act_sp):
        super(Q_net, self).__init__()
        self.act_sp = act_sp
        self.ob_sp = ob_sp
        self.affine1 = nn.Linear(self.ob_sp, 60)
        self.affine2 = nn.Linear(60 + self.act_sp, 60)
        self.value_head = nn.Linear(60, 1)

    def forward(self, x_, action_):
        x, action = x_.to(device).float(), action_.to(device).float()
        x = F.relu(self.affine1(x))
        x = torch.cat([x, action], 1)
        x = F.relu(self.affine2(x))
        return self.value_head(x)
