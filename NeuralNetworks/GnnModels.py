import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralNetworks.sublayer import MultiHeadAttention, PositionwiseFeedForward
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv


class NewGAT(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(NewGAT, self).__init__()
        self.slf_attn = GATv2Conv(output_dim, output_dim // nheads, nheads)
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, edge_index):
        x = x.squeeze()
        x = self.slf_attn(x, edge_index)
        x = x.unsqueeze(0)
        x = self.pos_ffn(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, output_dim, nheads, dropout=0):
        super(GATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, output_dim // 4, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, dropout=dropout)

    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x


class SmaacActor(nn.Module):
    def __init__(self, output_dim, nheads, node, action_dim, dropout=0, init_w=3e-3, log_std_min=-10, log_std_max=1):
        super(SmaacActor, self).__init__()
        self.gat1 = GATLayer(output_dim, nheads, dropout)
        self.gat2 = GATLayer(output_dim, nheads, dropout)
        self.gat3 = GATLayer(output_dim, nheads, dropout)
        self.down = nn.Linear(output_dim, 1)
        concat_dim = node * 2
        self.mu = nn.Linear(concat_dim, action_dim)
        self.log_std = nn.Linear(concat_dim, action_dim)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, x, adj):
        x, t = x
        x = self.gat1(x, adj)
        x = self.gat2(x, adj)
        x = self.gat3(x, adj)
        x = self.down(x).squeeze(-1)
        x = torch.cat([x, t], dim=-1)
        state = x
        x = F.leaky_relu(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std, state


class OrderActor(nn.Module):
    def __init__(self, node, action_dim, n_sub, log_std_min=-10, log_std_max=1):
        super(OrderActor, self).__init__()
        concat_dim = node * 2
        self.order_mu = nn.Linear(concat_dim + action_dim, n_sub)
        self.order_log_std = nn.Linear(concat_dim + action_dim, n_sub)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, state, action):
        state = torch.tanh(state)
        s_a = torch.cat([state, action], dim=1)
        order = self.order_mu(s_a)
        order_log_std = self.order_log_std(s_a)
        order_log_std = torch.clamp(order_log_std, self.log_std_min, self.log_std_max)
        return order, order_log_std


class SmaacSoftQ(nn.Module):
    def __init__(self, state_dim, nheads, node, action_dim, dropout=0, init_w=3e-3):
        super(SmaacSoftQ, self).__init__()
        self.gat1 = GATLayer(state_dim, nheads, dropout)
        self.down = nn.Linear(state_dim, 1)
        concat_dim = int(node + action_dim)
        hidden_dim = concat_dim // 4
        self.out = nn.Linear(concat_dim, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a, adj):
        x = self.gat1(x, adj)
        x = self.down(x).squeeze(-1)  # B,N
        x = torch.cat([x, a], dim=1)
        # if order is not None:
        #     x = torch.cat([x, order], dim=-1)
        x = F.leaky_relu(self.out(x))
        x = self.out2(x)
        return x


class SmaacDoubleSoftQ(nn.Module):
    def __init__(self, output_dim, nheads, node, action_dim, dropout=0, init_w=3e-3):
        super(SmaacDoubleSoftQ, self).__init__()
        self.Q1 = SmaacSoftQ(output_dim, nheads, node, action_dim, dropout, init_w)
        self.Q2 = SmaacSoftQ(output_dim, nheads, node, action_dim, dropout, init_w)

    def forward(self, x, a, adj):
        q1 = self.Q1(x, a, adj)
        q2 = self.Q2(x, a, adj)
        return q1, q2

    def min_Q(self, x, a, adj):
        q1, q2 = self.forward(x, a, adj)
        return torch.min(q1, q2)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nheads, node, dropout=0, num_layers=6):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gat_layers = nn.ModuleList()
        for l_idx in range(num_layers):
            self.gat_layers.append(GATLayer(output_dim, nheads, dropout))

    def forward(self, x, adj):
        x = self.linear(x)
        for l in self.gat_layers:
            x = l(x, adj)
        return x


class SoftQ(nn.Module):
    def __init__(self, state_dim, nheads, node_num, action_dim, dropout=0, num_layers=1):
        super(SoftQ, self).__init__()
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            self.layers.append(GATLayer(state_dim, nheads, dropout))
        self.down = nn.Linear(state_dim, 1)
        hidden_dim = node_num // 4
        self.out = nn.Linear(node_num, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, adj):
        for l in self.layers:
            x = l(x, adj)
        x = self.down(x).squeeze(-1)  # B,N
        x = F.leaky_relu(self.out(x))
        x = self.out2(x)
        return x


class DoubleSoftQEmb(nn.Module):
    def __init__(self, output_dim, nheads, node_num, action_dim, dropout=0, num_layers=1):
        super(DoubleSoftQEmb, self).__init__()
        self.Q1 = SoftQ(output_dim, nheads, node_num, action_dim, dropout, num_layers=num_layers)
        self.Q2 = SoftQ(output_dim, nheads, node_num, action_dim, dropout, num_layers=num_layers)

    def forward(self, x, adj):
        q1 = self.Q1(x, adj)
        q2 = self.Q2(x, adj)
        return q1, q2

    def min_Q(self, x, adj):
        q1, q2 = self.forward(x, adj)
        return torch.min(q1, q2)


class DoubleSoftQ(DoubleSoftQEmb):
    def __init__(self, input_dim, state_dim, nheads, node_num, action_dim, dropout=0, num_layers=3):
        super().__init__(state_dim, nheads, node_num, action_dim, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, state_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        return super().forward(x, adj)


class ActorEmb(nn.Module):
    def __init__(self, output_dim, nheads, node, action_dim, dropout=0, num_layers=3):
        super(ActorEmb, self).__init__()
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            self.layers.append(GATLayer(output_dim, nheads, dropout))
        self.down = nn.Linear(output_dim, 1)
        concat_dim = node * 2
        self.mu = nn.Linear(concat_dim, action_dim)

    def forward(self, x, adj):
        x, t = x
        for l in self.layers:
            x = l(x, adj)
        x = self.down(x).squeeze(-1)
        x = torch.cat([x, t], dim=-1)
        x = F.leaky_relu(x)
        mu = self.mu(x)
        probs = mu.softmax(dim=1)
        return probs


class Actor(ActorEmb):
    def __init__(self, input_dim, output_dim, nheads, node, action_dim, dropout=0, num_layers=3):
        super().__init__(output_dim, nheads, node, action_dim, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x, t = x
        x = [self.linear(x), t]
        return super().forward(x, adj)


class CriticNetwork(nn.Module):
    """
    This NN is used for the algorithms DQN and the critic in PPO
    """

    def __init__(self, input_dim, state_dim, nheads, node_num, action_dim=1, dropout=0, num_layers=3):
        super(CriticNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, state_dim)
        self.Q = SoftQ(state_dim, nheads, node_num, action_dim, dropout=dropout, num_layers=num_layers)

    def forward(self, x, adj):
        x = self.linear(x)
        return self.Q(x, adj)


class DeepQNetwork2(nn.Module):
    """
    Using the new implementation of GAT
    """

    def __init__(self, input_dim, state_dim, nheads, node_num, action_dim, dropout=0, num_layers=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, state_dim)
        self.layers = nn.ModuleList()
        for l_idx in range(num_layers):
            self.layers.append(NewGAT(state_dim, nheads, dropout))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(NewGAT(state_dim, nheads, dropout))
        self.down = nn.Linear(state_dim, 1)
        hidden_dim = node_num // 4
        self.out = nn.Linear(node_num, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.down.weight)
        gain = nn.init.calculate_gain("leaky_relu")
        nn.init.xavier_uniform_(self.out.weight, gain=gain)
        nn.init.xavier_uniform_(self.out2.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear(x)
        for l in self.layers:
            if isinstance(l, NewGAT):
                x = l(x, edge_index)
            else:
                x = l(x)
        x = self.down(x).squeeze(-1)  # B,N
        x = F.leaky_relu(self.out(x))
        x = self.out2(x)
        return x
