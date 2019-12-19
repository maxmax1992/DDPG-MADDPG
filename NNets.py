import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Policy_net(nn.Module):
    def __init__(self, ob_sp, act_sp):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(ob_sp, 400)
        self.affine2 = nn.Linear(400, 300)
        self.mean_head = nn.Linear(300, act_sp)

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
        self.affine1 = nn.Linear(self.ob_sp, 400)
        self.affine2 = nn.Linear(400 + self.act_sp, 300)
        self.value_head = nn.Linear(300, 1)

    def forward(self, x_, action_):
        x, action = x_.to(device).float(), action_.to(device).float()
        x = F.relu(self.affine1(x))
        x = torch.cat([x, action], 1)
        x = F.relu(self.affine2(x))
        return self.value_head(x)
    