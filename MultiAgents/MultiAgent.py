import torch
from grid2op.Action import BaseAction

from Agents.l2rpn_base_agent import L2rpnAgent
from Agents.PPO import BasePPO
from Agents.SACD import BaseSacd, SacdShared
from MultiAgents.MASACD import DependentSacd
from MultiAgents.MAPPO import DependentPPO
from MultiAgents.MAconverters import MADiscActionConverter
from MultiAgents.MiddleAgent import RuleBasedSubPicker, RandomOrderedSubPicker, FixedSubPicker

AGENT = {
    "isacd_base": BaseSacd,
    "isacd_emb": SacdShared,
    "ippo": BasePPO,
    "dsacd_emb": DependentSacd,
    "dppo": DependentPPO,
    # "sharedsacd": BaseSacdSharedLayer,
}

MIDDLE_AGENT = {
    "fixed": FixedSubPicker,
    "random": RandomOrderedSubPicker,
    "capa": RuleBasedSubPicker,
}


class IMARL(L2rpnAgent):
    """
    Multi Agent for L2RPN.
    Each agent is responsible for one substation.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        middle_agent = MIDDLE_AGENT[kwargs.get("middle_agent")]
        self.sub_picker = middle_agent(self.action_converter.masked_sorted_sub, action_space=self.action_space)

        # create deep learning part of the agent
        self.create_DLA(**kwargs)

    def create_action_converter(self, env, mask, mask_hi, **kwargs):
        # use different action converter
        # act_conv =  MADiscActionConverter(env, mask, mask_hi)
        # self.n_agents = len(act_conv.n_sub_actions)
        return MADiscActionConverter(env, mask, mask_hi)

    def create_DLA(self, **kwargs):
        agent_type = AGENT[kwargs.get("agent")]
        agents = [
            agent_type(self.input_dim, act_dim, self.node_num, **kwargs)
            for act_dim in self.action_converter.n_sub_actions
        ]
        self.agents = dict(zip(self.action_converter.masked_sorted_sub, agents))

    def reset(self, obs):
        super().reset(obs)
        self.sub_picker.complete_reset()
        for agent in self.agents.values():
            agent.reset(obs)

    def cache_stat(self):
        cache = super().cache_stat()
        if self.goal is not None:
            (sub_2_act, action) = self.goal
            sub_vals = self.agents[sub_2_act].cache_stat()
            cache_extra = {
                "sub_vals": sub_vals,
            }
            cache.update(cache_extra)
        return cache

    def load_cache_stat(self, cache):
        super().load_cache_stat(cache)
        if self.goal is not None:
            (sub_2_act, action) = self.goal
            sub_vals = cache["sub_vals"]
            self.agents[sub_2_act].load_cache_stat(sub_vals)

    def agent_act(self, obs, is_safe, sample, dn_count=0) -> BaseAction:
        # generate action if not safe
        if not is_safe or (len(self.sub_picker.subs_2_act) > 0):
            with torch.no_grad():
                stacked_state = self.get_current_state().to(self.device)
                adj = self.adj.unsqueeze(0)
                sub_2_act = self.sub_picker.pick_sub(obs, sample)
                action = self.agents[sub_2_act].produce_action(stacked_state, adj, sample=sample)
                goal = (sub_2_act, action)
                act = self.action_converter.plan_act(goal, obs.topo_vect)
                if sample:
                    self.update_goal(goal)
                elif (act == self.action_space()) & (dn_count < len(self.sub_picker.masked_sorted_sub)):
                    # skip DoNothing action when we are not training
                    act = self.agent_act(obs, is_safe, sample, dn_count=dn_count + 1)
                return act
        else:
            return self.action_space()

    def save_start_transition(self):
        super().save_start_transition()
        sub, action = self.start_goal
        self.agents[sub].save_start_transition()

    def save_transition(self, reward, done, n_step=1):
        self.agent_step += 1
        next_state = self.get_current_state()
        next_adj = self.adj.clone()
        sub, action = self.start_goal
        self.agents[sub].save_transition(
            self.start_state,
            self.start_adj,
            action,
            reward,
            next_state,
            next_adj,
            int(done),
            n_step,
        )

    def check_start_update(self):
        agent_mem_sizes = [len(agent.memory) for agent in self.agents.values()]
        if max(agent_mem_sizes) >= self.update_start:
            return True
        return False

    def update(self):
        self.update_step += 1
        for agent in self.agents.values():
            if len(agent.memory) >= self.update_start:
                agent.update()

    def print_updates_per_agent(self):
        for agent in self.agents.items():
            print(f"Agent for sub {agent[0]} had {agent[1].update_step} updates")
        print(f"Middle-Agent transition matrix:\n{self.sub_picker.count}")

    def save_model(self, path, name):
        [agent.save_model(f"{path}", f"{name}_sub_{sub}") for sub, agent in self.agents.items()]

    def load_model(self, path, name=None):
        [agent.load_model(f"{path}", f"{name}_sub_{sub}") for sub, agent in self.agents.items()]


class DepMARL(IMARL):
    def update(self):
        self.update_step += 1
        transition_probs = self.sub_picker.transition_probs
        for agent, trans_probs_agent in zip(self.agents.values(), transition_probs):
            if len(agent.memory) >= self.update_start:
                agent.dependent_update(self.agents.values(), trans_probs_agent)


# class IMARLSharedLayer(IMARL):
#     """
#         Multi Agent with shared layer.
#         each agent is responsible for one substation.
#     """
#     def __init__(self, env, **kwargs):
#         self.state_dim = kwargs.get('state_dim', 128)
#         self.nheads = kwargs.get('head_number', 8)
#         self.dropout = kwargs.get('dropout', 0.)
#         self.embed_lr = kwargs.get('embed_lr', 5e-5)
#         super().__init__(env, **kwargs)
#
#     def create_DLA(self, **kwargs):
#         super().create_DLA(**kwargs)
#         self.shared_layer = EncoderLayer(self.input_dim, self.state_dim, self.nheads, self.node_num,
#                                 self.dropout, num_layers=kwargs.get('n_layers', 3)).to(self.device)
#
#         self.shared_layer.optimizer = optim.Adam(self.shared_layer.parameters(), lr=self.embed_lr)
#         self.shared_layer.eval()
#
#     def agent_act(self, obs, is_safe, sample, dn_count=0) -> BaseAction:
#         # generate action if not safe
#         if not is_safe or (len(self.sub_picker.subs_2_act) > 0):
#             with torch.no_grad():
#                 stacked_state = self.get_current_state().to(self.device)
#                 adj = self.adj.unsqueeze(0)
#                 sub_2_act = self.sub_picker.pick_sub(obs, sample)
#                 self.need4reset = True
#                 # produce action using the shared_layer as common layer.
#                 action = self.agents[sub_2_act].produce_action(stacked_state, adj, sample=sample,
#                                                                shared_layer=self.shared_layer)
#                 goal = (sub_2_act, action)
#                 if sample:
#                     self.update_goal(goal)
#                 return self.action_converter.plan_act(goal, obs.topo_vect)
#         else:
#             return self.action_space()
#
#     def update(self):
#         self.update_step += 1
#         for agent in self.agents.values():
#             if len(agent.memory) >= self.update_start:
#                 self.shared_layer = agent.update(shared_layer=self.shared_layer)
#
#     def save_model(self, path, name):
#         torch.save(self.shared_layer.state_dict(), os.path.join(path, f'{name}_emb.pt'))
#         [agent.save_model(f'{path}', f'{name}_sub_{sub}') for sub, agent in self.agents.items()]
#
#     def load_model(self, path, name=None):
#         head = ''
#         if name is not None:
#             head = name + '_'
#         emb = torch.load(os.path.join(path, f'{head}emb.pt'), map_location=self.device)
#         self.shared_layer.load_state_dict(emb)
#         [agent.load_model(f'{path}', f'{name}_sub_{sub}') for sub, agent in self.agents.items()]

# class ImprovedIMARL(IMARL):
#     def __init__(self, env, **kwargs):
#         super().__init__(env, **kwargs)
#         self.start_goal = {sub: None for sub in self.agents}
#         self.start_state = {sub: None for sub in self.agents}
#         self.start_adj = {sub: None for sub in self.agents}
#
#     def save_start_transition(self):
#         sub, action = self.goal
#         self.agents[sub].save_start_transition()
#         self.start_goal[sub] = self.goal
#         self.start_state[sub] = self.get_current_state()
#         self.start_adj[sub] = self.adj.clone()
#
#     def save_transition(self, reward, done, n_step=1):
#         self.agent_step += 1
#         next_state = self.get_current_state()
#         next_adj = self.adj.clone()
#         sub, action = self.start_goal
#         self.agents[sub].save_transition(self.start_state, self.start_adj, action, reward,
#                                          next_state, next_adj, int(done), n_step)
#         if done:
#             self.sub_picker.complete_reset()
