import torch

from l2rpn_base_agent import L2rpnAgent
from models import DeepQNetwork, DeepQNetwork2
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch as T
import os

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class DQN(L2rpnAgent):
    def create_DLA(self, **kwargs):
        super().create_DLA(**kwargs)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.eps_min = kwargs.get('eps_end', 0.05)
        self.eps_dec = kwargs.get('eps_dec', 5e-4)
        self.Q = DeepQNetwork(self.input_dim, self.state_dim, self.nheads, self.node_num, self.action_dim,
                             self.dropout).to(self.device)
        # optimizers
        self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)
        self.Q.eval()

    def agent_act(self, obs, is_safe, sample):
        if not is_safe:
            if sample & (np.random.rand() < self.epsilon):
                goal = T.randint(self.action_dim, ())
            else:
                stacked_state = self.get_current_state().to(self.device)
                stacked_t, stacked_x = stacked_state[..., -1:], stacked_state[..., :-1]
                adj = self.adj.unsqueeze(0)
                q_values = self.get_q_values(stacked_x, adj)
                goal = q_values.argmax()
            if sample:
                self.update_goal(goal)
            return self.action_converter.plan_act(goal, obs.topo_vect)
        else:
            return self.action_space()

    def get_q_values(self, stacked_x, adj, is_batch=False):
        return self.Q.forward(stacked_x, adj)

    def update(self):
        stacked_t, stacked_x, adj, actions, rewards, \
        stacked2_t, stacked2_x, adj2, dones, steps = super().update()

        self.Q.train()

        # Collect q_values of actions we took
        q_values = self.get_q_values(stacked_x, adj, is_batch=True)[range(self.batch_size), actions]
        q_next = self.get_q_values(stacked2_x, adj2, is_batch=True)
        q_targets = rewards + (1.0 - dones) * self.gamma**steps * T.max(q_next, dim=1)[0].unsqueeze(-1)

        # Stochastic Gradient Descent step
        loss = F.mse_loss(q_values.detach(), q_targets.squeeze()).to(self.device)
        self.Q.optimizer.zero_grad()  # apparently this is needed in PyTorch
        loss.backward()
        self.Q.optimizer.step()
        self.Q.eval()

        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

        return loss.data.numpy()

    def save_model(self, path, name):
        T.save(self.Q.state_dict(), os.path.join(path, f'{name}_Q.pt'))

    def load_model(self, path, name=None):
        head = ''
        if name is not None:
            head = name + '_'
        self.Q.load_state_dict(T.load(os.path.join(path, f'{head}Q.pt'), map_location=self.device))


class DQN2(DQN):
    def create_DLA(self, **kwargs):
        super().create_DLA(**kwargs)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.eps_min = kwargs.get('eps_end', 0.05)
        self.eps_dec = kwargs.get('eps_dec', 5e-4)
        self.Q = DeepQNetwork2(self.input_dim, self.state_dim, self.nheads, self.node_num, self.action_dim,
                             self.dropout).to(self.device)
        # optimizers
        self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)
        self.Q.eval()

    def get_q_values(self, stacked_x, adj, is_batch=False):
        # change to edge_index data
        if is_batch:
            graph_data = [
                Data(x=stacked_x[i], edge_index=a.squeeze().nonzero().t().contiguous(), num_nodes=self.node_num) for
                (i, a) in enumerate(adj.split(1))]
            q_values = T.zeros(self.batch_size, self.action_dim)
            for i, data in enumerate(graph_data):
                q_values[i] = self.Q(data)
            return q_values
        else:
            graph_data = Data(x=stacked_x, edge_index=adj.squeeze().nonzero().t().contiguous(), num_nodes=self.node_num)
            return self.Q.forward(graph_data)
