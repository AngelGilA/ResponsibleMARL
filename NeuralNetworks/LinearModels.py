import torch
import torch.nn as nn
import torch.nn.functional as F


def lin_block(dim_in, dim_out, *args, **kwargs):
    return nn.Sequential(nn.Linear(dim_in, dim_out, *args, **kwargs), nn.ReLU())


class Actor(nn.Module):
    """
    This neural network uses linear layers instead of the GAT
    """

    def __init__(self, input_dim, hidden_dim, node, action_dim, num_layers=3):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim for _ in range(num_layers)]
        self.layers = nn.ModuleList([lin_block(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self.down = nn.Linear(hidden_dim, 1)
        concat_dim = node * 2
        self.mu = nn.Linear(concat_dim, action_dim)

    def forward(self, x, adj):
        x, t = x
        for l in self.layers:
            x = l(x)
        x = self.down(x).squeeze(-1)
        x = torch.cat([x, t], dim=-1)
        x = F.leaky_relu(x)
        mu = self.mu(x)
        probs = mu.softmax(dim=1)
        return probs


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, node_num, action_dim=1, num_layers=3):
        super(CriticNetwork, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim for _ in range(num_layers)]
        self.layers = nn.ModuleList([lin_block(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self.down = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(node_num, node_num // 4)
        self.out2 = nn.Linear(node_num // 4, action_dim)

    def forward(self, x, adj=None):
        for l in self.layers:
            x = l(x)
        x = self.down(x).squeeze(-1)  # B,N
        x = F.leaky_relu(self.out(x))
        x = self.out2(x)
        return x


class DoubleSoftQ(nn.Module):
    def __init__(self, input_dim, hidden_dim, node_num, action_dim, num_layers=3):
        super(DoubleSoftQ, self).__init__()
        self.Q1 = CriticNetwork(input_dim, hidden_dim, node_num, action_dim=action_dim, num_layers=num_layers)
        self.Q2 = CriticNetwork(input_dim, hidden_dim, node_num, action_dim=action_dim, num_layers=num_layers)

    def forward(self, x, adj):
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2

    def min_Q(self, x, adj):
        q1, q2 = self.forward(x, adj)
        return torch.min(q1, q2)
