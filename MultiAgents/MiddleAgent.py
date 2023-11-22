import numpy as np
import random


class FixedSubPicker(object):
    def __init__(self, masked_sorted_sub, **kwargs):
        self.masked_sorted_sub = masked_sorted_sub
        self.subs_2_act = []
        n_subs = len(masked_sorted_sub)
        self.count = np.zeros((n_subs, n_subs), int)
        self.previous_sub = -1

    def complete_reset(self):
        self.subs_2_act = []
        self.previous_sub = -1

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act

    def count_transitions(self, next_sub):
        if self.previous_sub >= 0:
            prev = np.flatnonzero(self.masked_sorted_sub == self.previous_sub).squeeze()
            next = np.flatnonzero(self.masked_sorted_sub == next_sub).squeeze()
            self.count[prev, next] += 1

    @property
    def transition_probs(self):
        row_sums = self.count.sum(axis=1, keepdims=True)
        non_zero_rows = (row_sums != 0).squeeze()
        probs = np.zeros_like(self.count, float)
        probs[non_zero_rows] = self.count[non_zero_rows] / row_sums[non_zero_rows]
        return probs


class RandomOrderedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, **kwargs):
        super().__init__(masked_sorted_sub)

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # randomize the order in which the substations are activated
            self.subs_2_act = list(self.masked_sorted_sub)
            random.shuffle(self.subs_2_act)
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act


class RuleBasedSubPicker(FixedSubPicker):
    def __init__(self, masked_sorted_sub, action_space):
        super().__init__(masked_sorted_sub)
        self.sub_line_or = []
        self.sub_line_ex = []
        for sub in self.masked_sorted_sub:
            self.sub_line_or.append(
                np.flatnonzero(action_space.line_or_to_subid == sub)
            )
            self.sub_line_ex.append(
                np.flatnonzero(action_space.line_ex_to_subid == sub)
            )

    def pick_sub(self, obs, sample):
        if len(self.subs_2_act) == 0:
            # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
            # which applies an action to substations that require urgent care.
            rhos = []
            for sub in self.masked_sorted_sub:
                sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze()
                rho = np.append(
                    obs.rho[self.sub_line_or[sub_i]].copy(),
                    obs.rho[self.sub_line_ex[sub_i]].copy(),
                )
                rho[rho == 0] = 3
                rho_max = rho.max()
                rho_mean = rho.mean()
                rhos.append((rho_max, rho_mean))
            order = sorted(
                zip(self.masked_sorted_sub, rhos), key=lambda x: (-x[1][0], -x[1][1])
            )
            self.subs_2_act = list(list(zip(*order))[0])
        sub_2_act = self.subs_2_act.pop(0)
        if sample:
            self.count_transitions(sub_2_act)
            self.previous_sub = sub_2_act
        return sub_2_act
