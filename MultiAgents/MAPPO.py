import torch

from Agents.PPO import BasePPO


class DependentPPO(BasePPO):
    def dependent_update(self, all_agents, trans_probs):
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

        next_values = self.get_next_values(next_states, next_adj, all_agents, trans_probs)
        # Advantages
        advantages = self.compute_gae(values, rewards, dones, next_values, steps)
        self.update_critic_actor(advantages, states, adj, actions, log_probs, values, batches)

        self.memory.clear_memory()

    def get_next_values(self, next_states, next_adj, all_agents=None, trans_probs=None):
        with torch.no_grad():
            next_states = torch.cat(next_states, 0).to(self.device)
            state_x, state_t = next_states[..., :-1], next_states[..., -1:]
            next_adj = torch.stack(next_adj, 0).to(self.device)
            V_agent_t2 = torch.cat([agent.critic(state_x, next_adj) for agent in all_agents], 1)
            next_values = torch.Tensor(trans_probs) * V_agent_t2
        return next_values.sum(axis=1, keepdims=True)
