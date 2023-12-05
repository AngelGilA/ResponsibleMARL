import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from NeuralNetworks.GnnModels import SmaacDoubleSoftQ, SmaacActor
from converters import *
from grid2op.Agent import BaseAgent

from Agents.SACD import SacdGoal

EPSILON = 1e-6


class SAC(SacdGoal):
    """
    This is class contains the SMAAC-agent as described by Yoon et al.
    """

    def __init__(self, env, **kwargs):
        # order is depreciated, it will not work anymore...
        self.rule = kwargs.get("rule", "c")
        self.use_order = self.rule == "o"

        super().__init__(env, **kwargs)

    def create_action_converter(self, env, mask, mask_hi, bus_thresh=0.1):
        return ActionConverter(env, mask, mask_hi, bus_thresh=bus_thresh)

    def create_DLA(self, **kwargs):
        super().create_DLA()
        if self.use_order:
            import warnings

            warnings.warn("Dont use use_order!!! COMPLETELY REMOVED!")

    def create_critic_actor(self):
        self.Q = SmaacDoubleSoftQ(self.state_dim, self.nheads, self.node_num, self.action_dim, self.dropout).to(
            self.device
        )
        self.tQ = SmaacDoubleSoftQ(self.state_dim, self.nheads, self.node_num, self.action_dim, self.dropout).to(
            self.device
        )
        self.actor = SmaacActor(self.state_dim, self.nheads, self.node_num, self.action_dim, self.dropout).to(
            self.device
        )

    def def_target_entropy(self):
        self.target_entropy = -self.action_dim * 3
        self.log_alpha = torch.FloatTensor([-3]).to(self.device)
        self.log_alpha.requires_grad = True
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def produce_action(self, state, adj, reparameterize=False, sample=True):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        mu, log_std, state = self.actor(state, adj)
        if not sample:
            # use mean / mu instead of sampling
            action = torch.tanh(mu).squeeze(0)
            return action.detach().cpu()

        std = log_std.exp()
        normal = Normal(mu, std)
        if reparameterize:
            z = normal.rsample()
        else:
            z = normal.sample()
        action = torch.tanh(z).squeeze(0)

        if reparameterize:
            log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + EPSILON)
            # need to do sum, because pytorch needs scalar quantity for loss
            log_pi = log_pi.sum(1, keepdim=True)
            return action, log_pi

        return action.detach().cpu()

    def get_critic_loss(self, stacked_x, stacked_t, adj, actions, rewards, stacked2_x, stacked2_t, adj2, dones, steps):
        """
        implementation of soft actor critic (Haarnoja et al 2018)
        Soft actor critic implementation 'explained': https://www.youtube.com/watch?v=ioidsRlf79o
        """
        self.Q.train()
        self.emb.train()
        self.actor.eval()

        states = self.emb(stacked_x, adj)
        states2 = self.emb(stacked2_x, adj2)
        actor_input2 = [states2, stacked2_t.squeeze(-1)]
        with torch.no_grad():
            tstates2 = self.temb(stacked2_x, adj2).detach()
            action2, log_pi2 = self.produce_action(actor_input2, adj2, reparameterize=True)

            # Soft state value function:
            V_t2 = self.tQ.min_Q(tstates2, action2, adj2) - self.alpha * log_pi2
            # modified Bellman Backup for sof q-function
            Q_targets = rewards + (1 - dones) * self.gamma**steps * V_t2.detach()

        predQ1, predQ2 = self.Q(states, actions, adj)
        Q1_loss = F.mse_loss(predQ1, Q_targets)
        Q2_loss = F.mse_loss(predQ2, Q_targets)
        return Q1_loss, Q2_loss

    def get_actor_loss(self, stacked_x, stacked_t, adj):
        self.actor.train()
        states = self.emb(stacked_x, adj)
        actor_input = [states, stacked_t.squeeze(-1)]
        action, log_pi = self.produce_action(actor_input, adj, reparameterize=True)
        critic_input = action
        actor_loss = (self.alpha * log_pi - self.Q.min_Q(states, critic_input, adj)).mean()
        # alpha loss: Calculates the loss for the entropy temperature parameter.
        alpha_loss = self.log_alpha * (-log_pi.detach() - self.target_entropy).mean()
        return actor_loss, alpha_loss


class SMAAC(SAC):
    """
    This is class contains the SMAAC-agent as it was implemented by Yoon et al.
    NOTE: the big difference is that they generate a goal topology at the start of an episode even if it is in a safe state.
    This is undesirable (according to TSO experts)
    """

    def agent_act(self, obs, is_safe, sample) -> BaseAgent:
        # generate goal if it is initial or previous goal has been reached
        if self.goal is None or (not is_safe and self.low_len == -1):
            goal, low_actions = self.generate_goal(sample, obs, not sample)
            if len(low_actions) == 0:
                if (not sample) or (self.goal is None):
                    self.update_goal(goal, low_actions)
                return self.action_space()
            self.update_goal(goal, low_actions)

        act = self.pick_low_action(obs)
        return act

    def reconnect_line(self, obs):
        # if the agent can reconnect powerline not included in controllable substation, return action
        # otherwise, return None
        dislines = np.where(obs.line_status == False)[0]
        for i in dislines:
            act = None
            if obs.time_next_maintenance[i] != 0 and i in self.action_converter.lonely_lines:
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
