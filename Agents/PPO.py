import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
import numpy as np
from grid2op.Action import BaseAction

from Agents.l2rpn_base_agent import SingleAgent
from Agents.BaseAgent import MyBaseAgent
from NeuralNetworks.GnnModels import CriticNetwork, ActorEmb, Actor
from NeuralNetworks.LinearModels import CriticNetwork as LinCritic, Actor as LinActor


def create_critic_actor(
    input_dim, state_dim, nheads, node_num, action_dim, dropout, num_layers=3, network="gnn", expand_hidden_layer=False
):
    if network == "gnn":
        # define gnn for critic and actor
        critic = CriticNetwork(input_dim, state_dim, nheads, node_num, dropout=dropout, num_layers=num_layers)
        if expand_hidden_layer:
            actor = Actor(input_dim, state_dim, nheads, node_num, action_dim, dropout=dropout, num_layers=num_layers)
        else:
            actor = ActorEmb(input_dim, nheads, node_num, action_dim, dropout=dropout, num_layers=num_layers)
    elif network == "lin":
        critic = LinCritic(input_dim, state_dim, node_num, num_layers=num_layers)
        actor = LinActor(input_dim, state_dim, node_num, action_dim, num_layers=num_layers)
    else:
        raise Exception(f"{network} is not a valid network type.")
    return critic, actor


class PPOMemory:
    def __init__(self):
        self.states = []
        self.adj = []
        self.actions = []
        self.logprobs = []
        self.vals = []
        self.next_states = []
        self.next_adj = []
        self.rewards = []
        self.dones = []
        self.steps = []

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + batch_size] for i in batch_start]

        return (
            self.states,
            self.adj,
            self.actions,
            self.logprobs,
            self.vals,
            self.next_states,
            self.next_adj,
            self.rewards,
            self.dones,
            self.steps,
            batches,
        )

    def append(
        self,
        state,
        adj,
        action,
        log_prob,
        value,
        next_state,
        next_adj,
        reward,
        done,
        steps,
    ):
        self.states.append(state)
        self.adj.append(adj)
        self.actions.append(action)
        self.logprobs.append(log_prob)
        self.vals.append(value)
        self.next_states.append(next_state)
        self.next_adj.append(next_adj)
        self.rewards.append(reward)

        self.dones.append(done)
        self.steps.append(steps)

    def __len__(self):
        return len(self.rewards)

    def clear_memory(self):
        del self.states[:]
        del self.adj[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.vals[:]
        del self.next_states[:]
        del self.next_adj[:]
        del self.rewards[:]
        del self.dones[:]
        del self.steps[:]


class BasePPO(MyBaseAgent):
    def __init__(self, input_dim, action_dim, node_num, **kwargs):
        self.policy_clip = kwargs.get("epsilon", 0.2)
        self.gae_lambda = kwargs.get("lambda", 0.95)
        self.entropy_ceoff = kwargs.get("entropy", 0.001)
        super().__init__(input_dim, action_dim, node_num, **kwargs)

    def create_DLA(self, **kwargs):
        self.memory = PPOMemory()
        self.critic, self.actor = create_critic_actor(
            self.input_dim,
            self.state_dim,
            self.nheads,
            self.node_num,
            self.action_dim,
            self.dropout,
            num_layers=kwargs.get("n_layers", 3),
            network=kwargs.get("network", "gnn"),
        )
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.new_critic, self.new_actor = create_critic_actor(
            self.input_dim,
            self.state_dim,
            self.nheads,
            self.node_num,
            self.action_dim,
            self.dropout,
            num_layers=kwargs.get("n_layers", 3),
            network=kwargs.get("network", "gnn"),
        )
        self.new_actor.to(self.device)
        self.new_critic.to(self.device)
        # optimizers only for new_critic and actor
        self.critic.optimizer = optim.Adam(self.new_critic.parameters(), lr=self.critic_lr)
        self.actor.optimizer = optim.Adam(self.new_actor.parameters(), lr=self.actor_lr)

        self.actor.load_state_dict(self.new_actor.state_dict())
        self.critic.load_state_dict(self.new_critic.state_dict())

        self.critic.eval()
        self.actor.eval()
        self.new_critic.eval()
        self.new_actor.eval()

    def reset(self, obs):
        super().reset(obs)
        self.log_prob = None
        self.value = None

    def cache_stat(self):
        cache = super().cache_stat()
        cache_extra = {
            "log_prob": self.log_prob,
            "value": self.value,
        }
        cache.update(cache_extra)
        return cache

    def load_cache_stat(self, cache):
        super().load_cache_stat(cache)
        self.log_prob = cache["log_prob"]
        self.value = cache["value"]

    def produce_action(self, stacked_state, adj, learn=False, sample=True):
        """
        Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action
        """
        state_x, state_t = stacked_state[..., :-1], stacked_state[..., -1:]
        actor_input = [state_x, state_t.squeeze(-1)]
        action_probs = self.actor(actor_input, adj)
        if sample:
            action_probs = Categorical(action_probs.squeeze(0))
            action = action_probs.sample()
            # update current value and log_prob
            self.value = self.critic(state_x, adj)
            self.log_prob = action_probs.log_prob(action)
            return action
        else:
            return action_probs.argmax()  # TODO: use top N actions with prob

    def save_start_transition(self):
        self.start_log_prob = self.log_prob
        self.start_value = self.value

    def save_transition(self, start_state, start_adj, action, reward, next_state, next_adj, done, n_step):
        self.memory.append(
            start_state,
            start_adj,
            action,
            self.start_log_prob,
            self.start_value,
            reward,
            next_state,
            next_adj,
            int(done),
            n_step,
        )

    def get_next_values(self, next_states, next_adj):
        with torch.no_grad():
            next_states = torch.cat(next_states, 0).to(self.device)
            state_x, state_t = next_states[..., :-1], next_states[..., -1:]
            next_adj = torch.stack(next_adj, 0).to(self.device)
            next_values = self.critic(state_x, next_adj)
        return next_values

    def update(self):
        self.update_step += 1
        (
            states,
            adj,
            actions,
            log_probs,
            values,
            rewards,
            next_states,
            next_adj,
            dones,
            steps,
            batches,
        ) = self.memory.generate_batches(self.batch_size)

        next_values = self.get_next_values(next_states, next_adj)
        # Advantages
        advantages = self.compute_gae(values, rewards, dones, next_values, steps)
        self.update_critic_actor(advantages, states, adj, actions, log_probs, values, batches)

        self.memory.clear_memory()

    def update_critic_actor(self, advantages, states, adj, actions, log_probs, values, batches):
        states = torch.cat(states, 0).to(self.device)
        adj = torch.stack(adj, 0).to(self.device)
        actions = torch.stack(actions, 0).to(self.device)
        log_probs = torch.stack(log_probs, 0).to(self.device)
        values = torch.stack(values, 0).to(self.device)

        self.new_critic.train()
        self.new_actor.train()
        for batch in batches:
            b_states = states[batch, :]
            b_adj = adj[batch, :]
            b_actions = actions[batch]
            b_values = values[batch].squeeze()
            b_advantages = advantages[batch]
            old_logprobs = log_probs[batch]

            # Evaluate old actions and values using new policy
            critic_values, new_logprobs, dist_entropy = self.evaluate_action(b_states, b_adj, b_actions)

            # Importance ratio: p/q
            # prob_ratio = new_probs.exp() / old_probs.exp()
            prob_ratio = (new_logprobs - old_logprobs).exp()  # this is the same by the properties of exponents.

            # Actor loss using Surrogate loss
            surr1 = b_advantages * prob_ratio
            surr2 = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * b_advantages
            actor_loss = -torch.min(surr1, surr2)

            # Critic loss: critic loss - entropy
            returns = b_advantages + b_values
            critic_loss = 0.5 * F.mse_loss(returns, critic_values) - self.entropy_ceoff * dist_entropy

            # Total loss
            total_loss = (actor_loss + critic_loss).mean()

            # Backward gradients
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.actor.load_state_dict(self.new_actor.state_dict())
        self.critic.load_state_dict(self.new_critic.state_dict())

    def evaluate_action(self, state, adj, action):
        state_x, state_t = state[..., :-1], state[..., -1:]
        value = self.new_critic(state_x, adj)
        actor_input = [state_x, state_t.squeeze(-1)]
        action_probs = self.new_actor(actor_input, adj)
        action_probs = Categorical(action_probs.squeeze(0))
        log_prob = action_probs.log_prob(action)
        dist_entropy = action_probs.entropy()
        return value.squeeze(), log_prob, dist_entropy

    def compute_gae(self, values, rewards, dones, next_values, steps):
        # GAE: generalized advantage estimation
        advantage = torch.zeros(len(rewards))
        a_t = 0
        for t in range(len(rewards) - 1, -1, -1):
            a_t = (
                -values[t]
                + rewards[t]
                + (1 - dones[t]) * self.gamma ** steps[t] * (next_values[t] + self.gae_lambda * a_t)
            )
            advantage[t] = a_t
        return advantage.to(self.device)

    def save_model(self, path, name):
        torch.save(self.actor.state_dict(), os.path.join(path, f"{name}_actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"{name}_critic.pt"))

    def load_model(self, path, name=None):
        head = ""
        if name is not None:
            head = name + "_"
        pass
        self.new_actor.load_state_dict(torch.load(os.path.join(path, f"{head}actor.pt"), map_location=self.device))
        self.new_critic.load_state_dict(torch.load(os.path.join(path, f"{head}critic.pt"), map_location=self.device))

        self.actor.load_state_dict(self.new_actor.state_dict())
        self.critic.load_state_dict(self.new_critic.state_dict())


class PPO(SingleAgent, BasePPO):
    """
    Single agent PPO
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        BasePPO.__init__(self, self.input_dim, self.action_dim, self.node_num, **kwargs)

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

    def save_start_transition(self):
        super().save_start_transition()
        BasePPO.save_start_transition(self)

    def save_transition(self, reward, done, n_step=1):
        next_state, next_adj = super().save_transition(reward, done)
        BasePPO.save_transition(
            self, self.start_state, self.start_adj, self.start_goal, reward, next_state, next_adj, int(done), n_step
        )
