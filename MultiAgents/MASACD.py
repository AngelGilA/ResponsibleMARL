import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

from Agents.SACD import BaseSacd, SacdShared


class DependentSacd(SacdShared):
    def dependent_update(self, all_agents, trans_probs):
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
                all_agents,
                trans_probs,
            )
            self.update_critic(Q1_loss, Q2_loss)
            # actor loss
            actor_loss, alpha_loss = self.get_actor_loss(stacked_states, adj)
            self.update_actor(actor_loss, alpha_loss)
            if self.update_step % self.target_update == 0:
                # target update
                self.update_target_network(self.Q, self.tQ)
                self.update_target_network(self.emb, self.temb)

            self.emb.eval()
            self.last_mem_len = len(self.memory)

    def get_critic_loss(
        self,
        stacked_states,
        adj,
        actions,
        rewards,
        stacked_states2,
        adj2,
        dones,
        steps,
        all_agents=None,
        trans_probs=np.zeros(1),
    ):
        self.Q.train()
        self.emb.train()
        self.actor.eval()

        stacked2_t, stacked2_x = stacked_states2[..., -1:], stacked_states2[..., :-1]
        # compute Q-target values based on the received rewards.
        with torch.no_grad():
            # modified soft state-value calculation for discrete case
            V_agents_t2 = torch.zeros((self.batch_size, len(all_agents)))
            for i, agent in enumerate(all_agents):
                tstates2 = agent.temb(stacked2_x, adj2).detach()
                action_probs, log_action_probs = agent.produce_action(stacked_states2, adj2, learn=True)
                V_j_t2 = action_probs * (agent.tQ.min_Q(tstates2, adj2) - agent.alpha * log_action_probs)
                V_agents_t2[:, i] = V_j_t2.sum(dim=1)
            V_t2 = torch.Tensor(trans_probs) * V_agents_t2
            Q_targets = rewards + (1.0 - dones) * self.gamma**steps * (V_t2.sum(axis=1, keepdims=True))

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


# class BaseSacdSharedLayer():
#     def __init__(self, input_dim, action_dim, node_num, **kwargs):
#         super().__init__(input_dim, action_dim, node_num, **kwargs)
#
#     def create_DLA(self, **kwargs):
#         # NOTE: Removed embedded layer since this will be part of main agent.
#
#         self.create_critic_actor()
#         # copy parameters
#         self.tQ.load_state_dict(self.Q.state_dict())
#
#         # optimizers
#         self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)
#         self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
#
#         self.Q.eval()
#         self.tQ.eval()
#         self.actor.eval()
#
#     def produce_action(
#         self,
#         stacked_state,
#         adj,
#         learn=False,
#         sample=True,
#         shared_layer: EncoderLayer = None,
#     ):
#         """Given the state, produces an action, the probability of the action, the log probability of the action, and
#         the argmax action"""
#         # stacked_state # B, N, F
#         state_x, state_t = stacked_state[..., :-1], stacked_state[..., -1:]
#         state = shared_layer(state_x, adj).detach()
#         actor_input = [state, state_t.squeeze(-1)]
#
#         action_probs = self.actor(actor_input, adj)
#         if learn:
#             # Have to deal with situation of 0.0 probabilities because we can't do log 0
#             z = action_probs == 0.0
#             z = z.float() * 1e-8
#             log_action_probs = torch.log(action_probs + z)
#             return action_probs, log_action_probs
#         if sample:
#             return Categorical(action_probs.squeeze(0)).sample().cpu()
#         else:
#             return action_probs.argmax()
#
#     def update(self, shared_layer: EncoderLayer = None):
#         if len(self.memory) >= self.last_mem_len:
#             (
#                 stacked_states,
#                 adj,
#                 actions,
#                 rewards,
#                 stacked_states2,
#                 adj2,
#                 dones,
#                 steps,
#             ) = super(BaseSacd, self).update()
#
#             # critic loss
#             Q1_loss, Q2_loss = self.get_critic_loss(
#                 stacked_states,
#                 adj,
#                 actions,
#                 rewards,
#                 stacked_states2,
#                 adj2,
#                 dones,
#                 steps,
#                 shared_layer,
#             )
#             self.update_critic(Q1_loss, Q2_loss, shared_layer=shared_layer)
#             # actor loss
#             actor_loss, alpha_loss = self.get_actor_loss(stacked_states, adj, shared_layer=shared_layer)
#             self.update_actor(actor_loss, alpha_loss, shared_layer=shared_layer)
#             if self.update_step % self.target_update == 0:
#                 # target update
#                 self.update_target_network(self.Q, self.tQ)
#             self.last_mem_len = len(self.memory)
#         return shared_layer.eval()
#
#     def get_critic_loss(
#         self,
#         stacked_states,
#         adj,
#         actions,
#         rewards,
#         stacked_states2,
#         adj2,
#         dones,
#         steps,
#         shared_layer: EncoderLayer = None,
#     ):
#         self.Q.train()
#         shared_layer.train()
#         self.actor.eval()
#
#         stacked2_t, stacked2_x = stacked_states2[..., -1:], stacked_states2[..., :-1]
#         # compute Q-target values based on the received rewards.
#         with torch.no_grad():
#             tstates2 = shared_layer(stacked2_x, adj2).detach()
#             action_probs, log_action_probs = self.produce_action(
#                 stacked_states2, adj2, learn=True, shared_layer=shared_layer
#             )
#             # modified soft state-value calculation for discrete case
#             V_t2 = action_probs * (self.tQ.min_Q(tstates2, adj2) - self.alpha * log_action_probs)
#             V_t2 = V_t2.sum(dim=1).unsqueeze(-1)
#             Q_targets = rewards + (1.0 - dones) * self.gamma**steps * (V_t2)
#
#         # compute current Q-values
#         stacked_t, stacked_x = stacked_states[..., -1:], stacked_states[..., :-1]
#         states = shared_layer(stacked_x, adj)
#         predQ1, predQ2 = self.Q(states, adj)
#         predQ1 = predQ1.gather(1, actions.unsqueeze(1).long())
#         predQ2 = predQ2.gather(1, actions.unsqueeze(1).long())
#
#         # critic loss
#         Q1_loss = F.mse_loss(predQ1, Q_targets)
#         Q2_loss = F.mse_loss(predQ2, Q_targets)
#         return Q1_loss, Q2_loss
#
#     def update_critic(self, Q1_loss, Q2_loss, shared_layer: EncoderLayer = None):
#         loss = Q1_loss + Q2_loss
#         self.Q.optimizer.zero_grad()
#         shared_layer.optimizer.zero_grad()
#         loss.backward()
#         shared_layer.optimizer.step()
#         self.Q.optimizer.step()
#         self.Q.eval()
#
#     def get_actor_loss(self, stacked_states, adj, shared_layer: EncoderLayer = None):
#         """
#         Calculates the loss for the actor. This loss includes the additional entropy term
#         """
#         stacked_t, stacked_x = stacked_states[..., -1:], stacked_states[..., :-1]
#         self.actor.train()
#         states = shared_layer(stacked_x, adj)
#         action_probs, log_action_probs = self.produce_action(stacked_states, adj, learn=True, shared_layer=shared_layer)
#         inside_term = self.alpha * log_action_probs - self.Q.min_Q(states, adj)
#         actor_loss = (action_probs * inside_term).sum(dim=1).mean()
#         # log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)
#         # alpha loss: Calculates the loss for the entropy temperature parameter.
#         # Test: re-use action probabilities for temperature loss (from https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details_1)
#         alpha_loss = (
#             action_probs.detach() * (-self.log_alpha * (log_action_probs + self.target_entropy).detach())
#         ).mean()
#         return actor_loss, alpha_loss
#
#     def update_actor(self, actor_loss, alpha_loss, shared_layer: EncoderLayer = None):
#         shared_layer.optimizer.zero_grad()
#         self.actor.optimizer.zero_grad()
#         actor_loss.backward()
#         shared_layer.optimizer.step()
#         self.actor.optimizer.step()
#         self.actor.eval()
#         self.alpha_optim.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optim.step()
#         self.alpha = self.log_alpha.exp()
#
#     def save_model(self, path, name):
#         torch.save(self.actor.state_dict(), os.path.join(path, f"{name}_actor.pt"))
#         torch.save(self.Q.state_dict(), os.path.join(path, f"{name}_Q.pt"))
#
#     def load_model(self, path, name=None):
#         head = ""
#         if name is not None:
#             head = name + "_"
#         self.actor.load_state_dict(torch.load(os.path.join(path, f"{head}actor.pt"), map_location=self.device))
#         self.Q.load_state_dict(torch.load(os.path.join(path, f"{head}Q.pt"), map_location=self.device))
