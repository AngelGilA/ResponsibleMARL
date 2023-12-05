from abc import abstractmethod

import numpy as np
import torch
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction

from converters import ObsConverter, SimpleDiscActionConverter
from util import ReplayBuffer

EPSILON = 1e-6


class L2rpnAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        super(L2rpnAgent, self).__init__(env.action_space)
        mask = kwargs.get("mask", 2)
        mask_hi = kwargs.get("mask_hi", 19)
        self.danger = kwargs.get("danger", 0.9)
        self.thermal_limit = env._thermal_limit_a
        self.node_num = env.dim_topo

        self.action_converter = self.create_action_converter(
            env, mask, mask_hi, bus_thresh=kwargs.get("threshold", 0.1)
        )
        self.obs_converter = ObsConverter(env, self.danger, self.device, attr=kwargs.get("input"))

        self.update_step = 0
        self.agent_step = 0
        self.memlen = kwargs.get("memlen", int(1e5))
        self.batch_size = kwargs.get("batch_size", 128)
        self.update_start = self.batch_size * kwargs.get("update_start", 8)
        self.gamma = kwargs.get("gamma", 0.99)

        self.n_history = kwargs.get("n_history", 6)
        self.input_dim = self.obs_converter.n_feature * self.n_history
        self.fc_ts = kwargs.get("forecast", 0)
        if self.fc_ts:
            print(f"NOTE: Using forecast {self.fc_ts} time steps ahead in this model")
            self.input_dim += self.fc_ts

        # print(kwargs)

    def create_action_converter(self, env, mask, mask_hi, **kwargs) -> SimpleDiscActionConverter:
        # by default use simple discrete action converter
        return SimpleDiscActionConverter(env, mask, mask_hi)

    def is_safe(self, obs):
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True

    def load_mean_std(self, mean, std):
        self.state_mean = mean
        self.state_std = std.masked_fill(std < 1e-5, 1.0)
        self.state_mean[0, sum(self.obs_space.shape[:20]) :] = 0
        self.state_std[0, sum(self.action_space.shape[:20]) :] = 1

    def state_normalize(self, s):
        s = (s - self.state_mean) / self.state_std
        return s

    def reset(self, obs):
        self.obs_converter.last_topo = np.ones(self.node_num, dtype=int)
        self.topo = None
        self.goal = None
        self.adj = None
        self.stacked_obs = []
        self.forecast = []
        self.start_state = None
        self.start_adj = None
        self.save = False

    def cache_stat(self):
        cache = {
            "last_topo": self.obs_converter.last_topo,
            "topo": self.topo,
            "goal": self.goal,
            "adj": self.adj,
            "stacked_obs": self.stacked_obs,
            "forecast": self.forecast,
            "start_date": self.start_state,
            "start_adj": self.start_adj,
            "save": self.save,
        }
        return cache

    def load_cache_stat(self, cache):
        self.obs_converter.last_topo = cache["last_topo"]
        self.topo = cache["topo"]
        self.goal = cache["goal"]
        self.adj = cache["adj"]
        self.stacked_obs = cache["stacked_obs"]
        self.forecast = cache["forecast"]
        self.start_state = cache["start_date"]
        self.start_adj = cache["start_adj"]
        self.save = cache["save"]

    def hash_goal(self, goal):
        hashed = ""
        for i in goal.view(-1):
            hashed += str(int(i.item()))
        return hashed

    def stack_obs(self, obs):
        obs_vect = obs.to_vect()
        obs_vect = self.state_normalize(torch.FloatTensor(obs_vect).unsqueeze(0))
        obs_vect, self.topo = self.obs_converter.convert_obs(obs_vect)
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)
        # self.adj = (torch.FloatTensor(obs.connectivity_matrix())).to(self.device)
        self.adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)

        if self.fc_ts & (obs.current_step + self.fc_ts < obs.max_step):
            load_p, load_q, prod_p, prod_v, maintenance = obs.get_forecast_arrays()
            if len(self.forecast):
                new_fc = self.obs_converter.convert_fc(prod_p[-1], load_p[-1], self.state_mean, self.state_std)
                self.forecast.pop(0)
                self.forecast.append(new_fc)
            else:
                for prod, load in zip(prod_p[1:], load_p[1:]):
                    new_fc = self.obs_converter.convert_fc(prod, load, self.state_mean, self.state_std)
                    self.forecast.append(new_fc)
        self.obs_converter.last_topo = np.where(obs.topo_vect == -1, self.obs_converter.last_topo, obs.topo_vect)

    def get_current_state(self):
        return torch.cat(
            self.stacked_obs + self.forecast + [self.topo] if self.fc_ts else self.stacked_obs + [self.topo],
            dim=-1,
        )

    def act(self, obs, reward, done=False):
        sample = reward is None  # if reward is None we are TRAINING therefore take sample!
        self.stack_obs(obs)
        is_safe = self.is_safe(obs)
        self.save = False

        # reconnect powerline when the powerline is disconnected
        if False in obs.line_status:
            act = self.reconnect_line(obs)
            if act is not None:
                return act

        return self.agent_act(obs, is_safe, sample)

    def reconnect_line(self, obs):
        # if the agent can reconnect powerline not included in controllable substation, return action
        # otherwise, return None
        dislines = np.where(obs.line_status == False)[0]
        for i in dislines:
            act = None
            if obs.time_next_maintenance[i] != 0:  # REMOVED: check for lonely lines
                sub_or = self.action_space.line_or_to_subid[i]
                sub_ex = self.action_space.line_ex_to_subid[i]
                if obs.time_before_cooldown_sub[sub_or] == 0:
                    act = self.action_space({"set_bus": {"lines_or_id": [(i, 1)]}})
                if obs.time_before_cooldown_sub[sub_ex] == 0:
                    act = self.action_space({"set_bus": {"lines_ex_id": [(i, 1)]}})
                if obs.time_before_cooldown_line[i] == 0:
                    status = self.action_space.get_change_line_status_vect()
                    status[i] = True
                    act = self.action_space({"change_line_status": status})
                if act is not None:
                    return act
        return None

    @abstractmethod
    def agent_act(self, obs, is_safe, reward) -> BaseAction:
        pass

    def save_start_transition(self):
        # save everything currently in goal
        self.start_goal = self.goal.clone() if isinstance(self.goal, torch.Tensor) else self.goal
        self.start_state = self.get_current_state()
        self.start_adj = self.adj.clone()

    def update_goal(self, goal, **kwargs):
        self.goal = goal
        self.save = True

    @abstractmethod
    def save_transition(self, reward, done, n_step=1):
        self.agent_step += 1
        next_state = self.get_current_state()
        next_adj = self.adj.clone()
        return next_state, next_adj

    def check_start_update(self):
        return len(self.memory) >= self.update_start

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


class SingleAgent(L2rpnAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.action_dim = self.action_converter.n
        self.nheads = kwargs.get("head_number", 8)
        self.dropout = kwargs.get("dropout", 0.0)
        self.actor_lr = self.critic_lr = self.embed_lr = kwargs.get("lr", 5e-5)

        self.state_dim = kwargs.get("state_dim", 128)

        # print(f'N: {self.node_num}, O: {self.input_dim}, S: {self.state_dim}, A: {self.action_dim}')
        # create deep learning part of the agent

        # self.create_DLA(**kwargs)

    pass
