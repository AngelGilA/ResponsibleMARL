import torch
from abc import abstractmethod, ABC


class MyBaseAgent(ABC):
    def __init__(self, input_dim, action_dim, node_num, **kwargs):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.state_dim = kwargs.get("state_dim", 128)
        self.node_num = node_num
        self.update_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nheads = kwargs.get("head_number", 8)

        self.gamma = kwargs.get("gamma", 0.99)
        self.dropout = kwargs.get("dropout", 0.0)
        self.batch_size = kwargs.get("batch_size", 128)
        self.actor_lr = self.critic_lr = kwargs.get("lr", 5e-5)

        self.update_step = 0
        self.agent_step = 0

        # create deep learning part of the agent
        self.create_DLA(**kwargs)

    @abstractmethod
    def create_DLA(self, **kwargs):
        pass

    def reset(self, obs):
        pass

    def cache_stat(self):
        return {}

    def load_cache_stat(self, cache):
        pass

    def unpack_batch(self, batch):
        states, adj, actions, rewards, states2, adj2, dones, steps = list(zip(*batch))
        states = torch.cat(states, 0)
        states2 = torch.cat(states2, 0)
        adj = torch.stack(adj, 0)
        adj2 = torch.stack(adj2, 0)
        actions = torch.stack(actions, 0)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        steps = torch.FloatTensor(steps).unsqueeze(1)
        return (
            states.to(self.device),
            adj.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            states2.to(self.device),
            adj2.to(self.device),
            dones.to(self.device),
            steps.to(self.device),
        )

    @abstractmethod
    def produce_action(self, stacked_state, adj, learn=False, sample=True):
        pass

    def save_start_transition(self):
        pass

    @abstractmethod
    def save_transition(self, start_state, start_adj, action, reward, next_state, next_adj, done, n_step):
        pass

    @abstractmethod
    def save_model(self, path, name):
        pass

    @abstractmethod
    def load_model(self, path, name=None):
        pass
