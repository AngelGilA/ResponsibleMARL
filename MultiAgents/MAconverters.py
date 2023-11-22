import numpy as np

from converters import SimpleDiscActionConverter


class MADiscActionConverter(SimpleDiscActionConverter):
    def __init__(self, env, mask, mask_hi, rule="c"):
        super().__init__(env, mask, mask_hi, rule)
        self.n = 0  # not relevant for MA

    def init_action_converter(self):
        # sort subs descending:
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        self.masked_sorted_sub = self.subs[sort_subs]
        # TO DO: Decide if I want to design other sort rules

        self.sub_actions = []
        self.n_sub_actions = []
        self.sub_line_or = []
        self.sub_line_ex = []
        for sub in self.masked_sorted_sub:
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub)
            self.sub_actions.append(topo_actions)
            self.n_sub_actions.append(len(topo_actions))
            self.sub_line_or.append(np.flatnonzero(self.action_space.line_or_to_subid == sub))
            self.sub_line_ex.append(np.flatnonzero(self.action_space.line_ex_to_subid == sub))

    def plan_act(self, goal, topo_vect, **kwargs):
        sub, action = goal
        sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze()
        action = self.sub_actions[sub_i][action]
        # return action
        if self.inspect_act(sub, action, topo_vect):
            return action
        else:  # return do nothing action.
            return self.action_space()


class MADiscActionConverter2(MADiscActionConverter):
    """
    Only difference is the sorting rule which is ascending instead of DEscending.
    """

    def init_action_converter(self):
        # sort subs AScending:
        sort_subs = np.argsort(self.action_space.sub_info[self.action_space.sub_info > self.mask])
        self.masked_sorted_sub = self.subs[sort_subs]
        # TO DO: Decide if I want to design other sort rules

        self.sub_actions = []
        self.n_sub_actions = []
        self.sub_line_or = []
        self.sub_line_ex = []
        for sub in self.masked_sorted_sub:
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub)
            self.sub_actions.append(topo_actions)
            self.n_sub_actions.append(len(topo_actions))
            self.sub_line_or.append(np.flatnonzero(self.action_space.line_or_to_subid == sub))
            self.sub_line_ex.append(np.flatnonzero(self.action_space.line_ex_to_subid == sub))
