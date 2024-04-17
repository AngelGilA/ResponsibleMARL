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
            # print(prev, next)
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

    
class ClusteredCAPAPicker(object):
    def __init__(self, action_space, clusters):
        self.action_space = action_space
        self.clusters = clusters
        self.cluster_priority = []  # To store priority of each cluster
        self.activation_sequence = []  # Ordered list of clusters based on priority (equivalent to subs_2_act)
        self.previous_ra = -1
        n_ra = len(clusters)
        self.count = np.zeros((n_ra, n_ra), int)

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)
        # Sort clusters based on priority
        self.activation_sequence = sorted(range(len(self.clusters)), key=lambda i: -self.cluster_priority[i])
        #print("self.activation_sequece", self.activation_sequence)
        #print("self.cluster_priority", self.cluster_priority)
        #print("self.clusters", self.clusters)

    def calculate_cluster_priority(self, obs):
        self.cluster_priority = []
        for cluster in self.clusters:
            cluster_max_rho = max([max(obs.rho[self.action_space.line_or_to_subid == sub]) for sub in cluster if len(obs.rho[self.action_space.line_or_to_subid == sub]) > 0])
            self.cluster_priority.append(cluster_max_rho)
    
    def pick_cluster(self, obs, sample):
        if not self.activation_sequence:
            # If the activation sequence is empty, it means priorities need to be recalculated
            self.update_activation_sequence(obs)

        if sample:
            self.count_ra_transitions(self.activation_sequence[0])
            self.previous_ra = self.activation_sequence[0]
        return self.activation_sequence.pop(0)  # Pop the highest priority cluster
    
    def complete_reset(self):
        self.activation_sequence = []
        self.previous_ra = -1 # Used to avoid counting as transition initialization or reset.

    def count_ra_transitions(self, next_ra):
        if self.previous_ra >= 0:
            # print("\n ", self.clusters, self.previous_ra, next_ra)
            prev = self.previous_ra
            next = next_ra
            self.count[prev, next] += 1

    def count_sub_transitions(self, next_sub):
        pass

    # To do: transition probabilities

# still need to limit the amount of times in a row that it can access the same RA
class UrgentCAPAPicker(ClusteredCAPAPicker):
    def __init__(self, action_space, clusters):
        super(UrgentCAPAPicker, self).__init__(action_space, clusters) 

    def update_activation_sequence(self, obs):
        self.calculate_cluster_priority(obs)

        # Set activation sequence to only contain this most urgent cluster
        most_urgent_cluster_index = self.cluster_priority.index(max(self.cluster_priority))
        self.activation_sequence = [most_urgent_cluster_index]