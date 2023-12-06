import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import os
import numpy as np
from grid2op.Action import BaseAction

from converters import GoalActionConverter
from util import ReplayBuffer
from NeuralNetworks.GnnModels import DoubleSoftQ, Actor, EncoderLayer, DoubleSoftQEmb, ActorEmb
from NeuralNetworks.LinearModels import DoubleSoftQ as Lin2SoftQ, Actor as LinActor
from Agents.l2rpn_base_agent import SingleAgent
from Agents.BaseAgent import MyBaseAgent


class BaseSacd(MyBaseAgent):
    def __init__(self, input_dim, action_dim, node_num, **kwargs):
        self.target_update = kwargs.get("target_update", 1)
        self.target_entropy_scale = kwargs.get("target_entropy_scale", 0.98)
        self.tau = kwargs.get("tau", 1e-3)
        self.hard_target = kwargs.get("hard_target", False)
        self.target_update = kwargs.get("target_update", 1)
        self.embed_lr = self.alpha_lr = kwargs.get("lr", 5e-5)
        self.update_freq = kwargs.get("update_freq", 1)

        super().__init__(input_dim, action_dim, node_num, **kwargs)

        self.memlen = kwargs.get("memlen", int(1e5))
        self.memory = ReplayBuffer(max_size=self.memlen)
        # entropy
        self.def_target_entropy()
        self.last_mem_len = 0

    def create_DLA(self, **kwargs):
        self.create_critic_actor(num_layers=kwargs.get("n_layers", 3), network=kwargs.get("network", "gnn"))

        # copy parameters
        self.tQ.load_state_dict(self.Q.state_dict())

        # optimizers
        self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.Q.eval()
        self.tQ.eval()
        self.actor.eval()

    def create_critic_actor(self, num_layers=3, network="gnn"):
        if network == "gnn":
            # use different nn for critic and actor
            self.Q = DoubleSoftQ(
                self.input_dim,
                self.state_dim,
                self.nheads,
                self.node_num,
                self.action_dim,
                self.dropout,
                num_layers=num_layers,
            ).to(self.device)
            self.tQ = DoubleSoftQ(
                self.input_dim,
                self.state_dim,
                self.nheads,
                self.node_num,
                self.action_dim,
                self.dropout,
                num_layers=num_layers,
            ).to(self.device)
            self.actor = Actor(
                self.input_dim, self.state_dim, self.nheads, self.node_num, self.action_dim, num_layers=num_layers
            ).to(self.device)
        elif network == "lin":
            # use different nn for critic and actor
            self.Q = Lin2SoftQ(
                self.input_dim,
                self.state_dim,
                self.node_num,
                self.action_dim,
                num_layers=num_layers,
            ).to(self.device)
            self.tQ = Lin2SoftQ(
                self.input_dim,
                self.state_dim,
                self.node_num,
                self.action_dim,
                num_layers=num_layers,
            ).to(self.device)
            self.actor = LinActor(
                self.input_dim, self.state_dim, self.node_num, self.action_dim, num_layers=num_layers
            ).to(self.device)
        else:
            raise Exception(f"{network} is not a valid network type.")

    def def_target_entropy(self):
        # we set the max possible entropy as the target entropy
        self.target_entropy = -np.log((1.0 / self.action_dim)) * self.target_entropy_scale
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def produce_action(self, stacked_state, adj, learn=False, sample=True):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        # stacked_state # B, N, F (batches, elements, features)
        state_x, state_t = stacked_state[..., :-1], stacked_state[..., -1:]
        actor_input = [state_x, state_t.squeeze(-1)]

        action_probs = self.actor(actor_input, adj)
        if learn:
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_action_probs = torch.log(action_probs + z)
            return action_probs, log_action_probs
        if sample:
            return Categorical(action_probs.squeeze(0)).sample().cpu()
        else:
            return action_probs.argmax()

    def save_transition(self, start_state, start_adj, action, reward, next_state, next_adj, done, n_step):
        self.memory.append((start_state, start_adj, action, reward, next_state, next_adj, done, n_step))

    def update(self):
        if len(self.memory) >= self.last_mem_len + self.update_freq:
            self.update_step += 1
            (
                stacked_states,
                adj,
                actions,
                rewards,
                stacked_states2,
                adj2,
                dones,
                steps,
            ) = self.unpack_batch(self.memory.sample(self.batch_size))

            # critic loss
            Q1_loss, Q2_loss = self.get_critic_loss(
                stacked_states,
                adj,
                actions,
                rewards,
                stacked_states2,
                adj2,
                dones,
                steps,
            )
            self.update_critic(Q1_loss, Q2_loss)
            # actor loss
            actor_loss, alpha_loss = self.get_actor_loss(stacked_states, adj)
            self.update_actor(actor_loss, alpha_loss)
            if self.update_step % self.target_update == 0:
                # target update
                self.update_target_network(self.Q, self.tQ)

            self.last_mem_len = len(self.memory)

    def update_target_network(self, local_model, target_model):
        if self.hard_target:
            target_model.load_state_dict(local_model.state_dict())
        else:
            for tp, p in zip(target_model.parameters(), local_model.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def get_critic_loss(self, stacked_states, adj, actions, rewards, stacked_states2, adj2, dones, steps):
        self.Q.train()
        self.actor.eval()

        stacked2_x, stacked2_t = stacked_states2[..., :-1], stacked_states2[..., -1:]
        # compute Q-target values based on the received rewards.
        with torch.no_grad():
            action_probs, log_action_probs = self.produce_action(stacked_states2, adj2, learn=True)
            # modified soft state-value calculation for discrete case
            V_t2 = action_probs * (self.tQ.min_Q(stacked2_x, adj2) - self.alpha * log_action_probs)
            V_t2 = V_t2.sum(dim=1).unsqueeze(-1)
            Q_targets = rewards + (1.0 - dones) * self.gamma**steps * (V_t2)

        # compute current Q-values
        stacked_x, stacked_t = stacked_states[..., :-1], stacked_states[..., -1:]
        predQ1, predQ2 = self.Q(stacked_x, adj)
        predQ1 = predQ1.gather(1, actions.unsqueeze(1).long())
        predQ2 = predQ2.gather(1, actions.unsqueeze(1).long())

        # critic loss
        Q1_loss = F.mse_loss(predQ1, Q_targets)
        Q2_loss = F.mse_loss(predQ2, Q_targets)
        return Q1_loss, Q2_loss

    def update_critic(self, Q1_loss, Q2_loss):
        loss = Q1_loss + Q2_loss
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()
        self.Q.eval()

    def get_actor_loss(self, stacked_states, adj):
        """
        Calculates the loss for the actor. This loss includes the additional entropy term
        """
        stacked_x, stacked_t = stacked_states[..., :-1], stacked_states[..., -1:]
        self.actor.train()
        action_probs, log_action_probs = self.produce_action(stacked_states, adj, learn=True)
        inside_term = self.alpha * log_action_probs - self.Q.min_Q(stacked_x, adj)
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()
        # log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)
        # alpha loss: Calculates the loss for the entropy temperature parameter.
        # Test: re-use action probabilities for temperature loss (from https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details_1)
        alpha_loss = (
            action_probs.detach() * (-self.log_alpha * (log_action_probs + self.target_entropy).detach())
        ).mean()
        return actor_loss, alpha_loss

    def update_actor(self, actor_loss, alpha_loss):
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.eval()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    def save_model(self, path, name):
        torch.save(self.actor.state_dict(), os.path.join(path, f"{name}_actor.pt"))
        torch.save(self.Q.state_dict(), os.path.join(path, f"{name}_Q.pt"))

    def load_model(self, path, name=None):
        head = ""
        if name is not None:
            head = name + "_"
        self.actor.load_state_dict(torch.load(os.path.join(path, f"{head}actor.pt"), map_location=self.device))
        emb = torch.load(os.path.join(path, f"{head}emb.pt"), map_location=self.device)
        self.Q.load_state_dict(torch.load(os.path.join(path, f"{head}Q.pt"), map_location=self.device))


class SacdShared(BaseSacd):
    """
    Soft Actor Critic Discrete, similar to SacdSimple, but with a shared embedded layer, that is used by both the
    actor and the critic.
    """

    def create_DLA(self, **kwargs):
        super().create_DLA(**kwargs)
        num_lay = kwargs.get("n_layers", 3)
        self.emb = EncoderLayer(
            self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout, num_layers=num_lay
        ).to(self.device)
        self.temb = EncoderLayer(
            self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout, num_layers=num_lay
        ).to(self.device)

        # copy parameters
        self.temb.load_state_dict(self.emb.state_dict())

        # optimizers
        self.emb.optimizer = optim.Adam(self.emb.parameters(), lr=self.embed_lr)

        self.emb.eval()
        self.temb.eval()

    def create_critic_actor(self, num_layers=3, network="gnn"):
        if network == "gnn":
            # use different nn for critic and actor
            self.Q = DoubleSoftQEmb(self.state_dim, self.nheads, self.node_num, self.action_dim, self.dropout).to(
                self.device
            )
            self.tQ = DoubleSoftQEmb(self.state_dim, self.nheads, self.node_num, self.action_dim, self.dropout).to(
                self.device
            )
            self.actor = ActorEmb(
                self.state_dim, self.nheads, self.node_num, self.action_dim, num_layers=num_layers
            ).to(self.device)
        elif network == "lin":
            # use different nn for critic and actor
            self.Q = Lin2SoftQ(
                self.state_dim,
                self.state_dim,
                self.node_num,
                self.action_dim,
                num_layers=num_layers,
            ).to(self.device)
            self.tQ = Lin2SoftQ(
                self.state_dim,
                self.state_dim,
                self.node_num,
                self.action_dim,
                num_layers=num_layers,
            ).to(self.device)
            self.actor = LinActor(
                self.state_dim, self.state_dim, self.node_num, self.action_dim, num_layers=num_layers
            ).to(self.device)
        else:
            raise Exception(f"{network} is not a valid network type.")

    def produce_action(self, stacked_state, adj, learn=False, sample=True):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        # stacked_state # B, N, F (batches, elements, features)
        state_x, state_t = stacked_state[..., :-1], stacked_state[..., -1:]
        state = self.emb(state_x, adj).detach()
        actor_input = [state, state_t.squeeze(-1)]

        action_probs = self.actor(actor_input, adj)
        if learn:
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_action_probs = torch.log(action_probs + z)
            return action_probs, log_action_probs
        if sample:
            return Categorical(action_probs.squeeze(0)).sample().cpu()
        else:
            return action_probs.argmax()

    def update(self):
        if len(self.memory) >= self.last_mem_len + self.update_freq:
            super().update()

            if self.update_step % self.target_update == 0:
                # target update
                self.update_target_network(self.emb, self.temb)

            self.emb.eval()

    def get_critic_loss(self, stacked_states, adj, actions, rewards, stacked_states2, adj2, dones, steps):
        self.Q.train()
        self.emb.train()
        self.actor.eval()

        stacked2_x, stacked2_t = stacked_states2[..., :-1], stacked_states2[..., -1:]
        with torch.no_grad():
            tstates2 = self.temb(stacked2_x, adj2).detach()
            action_probs, log_action_probs = self.produce_action(stacked_states2, adj2, learn=True)
            # modified soft state-value calculation for discrete case
            V_t2 = action_probs * (self.tQ.min_Q(tstates2, adj2) - self.alpha * log_action_probs)
            V_t2 = V_t2.sum(dim=1).unsqueeze(-1)
            Q_targets = rewards + (1.0 - dones) * self.gamma**steps * (V_t2)

        # compute current Q-values
        stacked_t, stacked_x = stacked_states[..., -1:], stacked_states[..., :-1]
        states = self.emb(stacked_x, adj)
        predQ1, predQ2 = self.Q(states, adj)
        predQ1 = predQ1.gather(1, actions.unsqueeze(1).long())
        predQ2 = predQ2.gather(1, actions.unsqueeze(1).long())

        # critic loss
        Q1_loss = F.mse_loss(predQ1, Q_targets)
        Q2_loss = F.mse_loss(predQ2, Q_targets)
        return Q1_loss, Q2_loss

    def update_critic(self, Q1_loss, Q2_loss):
        loss = Q1_loss + Q2_loss
        self.Q.optimizer.zero_grad()
        self.emb.optimizer.zero_grad()
        loss.backward()
        self.emb.optimizer.step()
        self.Q.optimizer.step()
        self.Q.eval()

    def get_actor_loss(self, stacked_states, adj):
        """
        Calculates the loss for the actor. This loss includes the additional entropy term
        """
        stacked_x, stacked_t = stacked_states[..., :-1], stacked_states[..., -1:]
        self.actor.train()
        action_probs, log_action_probs = self.produce_action(stacked_states, adj, learn=True)
        states_x = self.emb(stacked_x, adj)
        inside_term = self.alpha * log_action_probs - self.Q.min_Q(states_x, adj)
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()
        # log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)
        # alpha loss: Calculates the loss for the entropy temperature parameter.
        # Test: re-use action probabilities for temperature loss (from https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details_1)
        alpha_loss = (
            action_probs.detach() * (-self.log_alpha * (log_action_probs + self.target_entropy).detach())
        ).mean()
        return actor_loss, alpha_loss

    def update_actor(self, actor_loss, alpha_loss):
        self.emb.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.emb.optimizer.step()
        self.actor.optimizer.step()
        self.actor.eval()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    def save_model(self, path, name):
        super().save_model(path, name)
        torch.save(self.emb.state_dict(), os.path.join(path, f"{name}_emb.pt"))

    def load_model(self, path, name=None):
        head = ""
        if name is not None:
            head = name + "_"
        super().load_model(path, name=name)
        self.emb.load_state_dict(torch.load(os.path.join(path, f"{head}emb.pt"), map_location=self.device))
        # copy parameters
        self.temb.load_state_dict(self.emb.state_dict())


class SacdSimple(SingleAgent, BaseSacd):
    """
    Soft Actor Critic Discrete, using the "simple action converter",
    This agent will only pick one action each ts and not for multiple substations at once.
    Furthermore, this SACD algorithm will not use a shared embedded layer
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        BaseSacd.__init__(self, self.input_dim, self.action_dim, self.node_num, **kwargs)

    def agent_act(self, obs, is_safe, sample) -> BaseAction:
        # generate action if not safe
        if not is_safe:
            with torch.no_grad():
                stacked_state = self.get_current_state().to(self.device)
                adj = self.adj.unsqueeze(0)
                goal = self.produce_action(stacked_state, adj, sample=sample)
                if sample:
                    self.update_goal(goal)
                return self.action_converter.plan_act(goal, obs.topo_vect)
        else:
            return self.action_space()

    def save_transition(self, reward, done, n_step=1):
        next_state, next_adj = super().save_transition(reward, done)
        BaseSacd.save_transition(
            self,
            self.start_state,
            self.start_adj,
            self.start_goal,
            reward,
            next_state,
            next_adj,
            int(done),
            n_step,
        )


class SacdEmb(SacdSimple, SacdShared):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        SacdShared.__init__(self, self.input_dim, self.action_dim, self.node_num, **kwargs)


class SacdGoal(SacdSimple):
    """
    Soft Actor Critic Discrete agent based on the SMAAC agent of Yoon et al.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.max_low_len = kwargs.get("max_low_len", 19)

    def create_action_converter(self, env, mask, mask_hi, **kwargs):
        # use different action converter
        return GoalActionConverter(env, mask, mask_hi)

    def reset(self, obs):
        super(SacdGoal, self).reset(obs)
        self.low_len = -1
        self.low_actions = []

    def cache_stat(self):
        cache = super().cache_stat()
        cache_extra = {
            "low_len": self.low_len,
            "low_actions": self.low_actions,
        }
        cache.update(cache_extra)
        return cache

    def load_cache_stat(self, cache):
        super(SacdGoal, self).load_cache_stat(cache)
        self.low_len = cache["low_len"]
        self.low_actions = cache["low_actions"]

    def agent_act(self, obs, is_safe, sample):
        # generate goal if it is initial or previous goal has been reached
        # REMOVED: start with generating goal topo if none exists
        # Reason -> This would result with starting each episode with a goal topology and is not how it is described in the paper
        # Furthermore, it is not desirable (according to TSO experts) to deviate from reference topology when this is not needed (no danger)
        if not is_safe and self.low_len == -1:
            goal, low_actions = self.generate_goal(sample, obs, not sample)
            # always update goal
            self.update_goal(goal, low_actions)

        act = self.pick_low_action(obs)
        return act

    def generate_goal(self, sample, obs, nosave=False):
        stacked_state = self.get_current_state().to(self.device)
        adj = self.adj.unsqueeze(0)
        goal = self.produce_action(stacked_state, adj, sample=sample)
        low_actions = self.action_converter.plan_act(goal, obs.topo_vect)
        return goal, low_actions

    def update_goal(self, goal, low_actions=None):
        super().update_goal(goal)
        self.low_actions = low_actions
        self.low_len = 0

    def pick_low_action(self, obs):
        # Safe and there is no queued low actions, just do nothing
        if self.is_safe(obs) and self.low_len == -1:
            act = self.action_space()
            return act

        # optimize low actions every step
        self.low_actions = self.action_converter.optimize_plan(obs, self.low_actions)
        self.low_len += 1

        # queue has been empty after optimization. just do nothing
        if len(self.low_actions) == 0:
            act = self.action_space()
        # normally execute low action from low actions queue
        else:
            sub_id, new_topo = self.low_actions.pop(0)[:2]
            act = self.action_converter.convert_act(sub_id, new_topo)

        # When it meets maximum low action execution time, log and reset
        if (len(self.low_actions) == 0) or (self.max_low_len <= self.low_len):
            self.low_len = -1
        return act
