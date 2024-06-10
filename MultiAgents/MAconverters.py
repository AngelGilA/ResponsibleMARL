import numpy as np

from converters import SimpleDiscActionConverter

from ResponsibilityAreas.GroupingMethods import perform_clustering

class MADiscActionConverter(SimpleDiscActionConverter):
    def __init__(self, env, mask, mask_hi, rule="c"):
        super().__init__(env, mask, mask_hi, rule)
        self.n = 0  # not relevant for MA

    def init_action_converter(self):
        # sort subs descending:
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        self.masked_sorted_sub = self.subs[sort_subs]
        #print(-self.action_space.sub_info[self.action_space.sub_info > self.mask]) [-6 -4 -6 -5 -7 -5 -4]
        #print(self.action_space.sub_info > self.mask)                   [False  True  True  True  True  True False False  True False
        #print(sort_subs)                                                [4 0 2 3 5 1 6]
        #print(self.masked_sorted_sub)                                   [ 5  1  3  4  8  2 12]
        # TO DO: Decide if I want to design other sort rules

        self.sub_actions = []
        self.n_sub_actions = []
        self.sub_line_or = []
        self.sub_line_ex = []
        # for cluster in clusters, take the actions of all its subs and store them in a list
        # what about sub_line_or and sub_line_ex? -> they are the indices of the lines that are connected to the substation
        # do we need to still store them per substation? or can it be done per cluster? explain doubt
        # where are each of these use and how? -> sub_actions: in plan_act, n_sub_actions: in plan_act, sub_line_or: in inspect_act, sub_line_ex: in inspect_act
         
        for sub in self.masked_sorted_sub:
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub)
            self.sub_actions.append(topo_actions)
            self.n_sub_actions.append(len(topo_actions))
            self.sub_line_or.append(np.flatnonzero(self.action_space.line_or_to_subid == sub))
            self.sub_line_ex.append(np.flatnonzero(self.action_space.line_ex_to_subid == sub))
        #print(self.sub_actions) ::list of lists:: <grid2op.Space.GridObjects.PlayableAction_l2rpn_case14_sandbox object at 0x000001AE251E6AD0>
        #print(self.n_sub_actions) [57, 29, 31, 15, 15, 5, 7]
        #print(self.sub_line_or)   [array([7, 8, 9], dtype=int64), array([2, 3, 4], dtype=int64), array([ 6, 15, 16], dtype=int64), ar
        #print(self.sub_line_ex)   [array([17], dtype=int64), array([0], dtype=int64), array([3, 5], dtype=int64), array([1, 4, 6], dtype=int64),
 
    def plan_act(self, goal, topo_vect, **kwargs):
        sub, action = goal
        sub_i = np.flatnonzero(self.masked_sorted_sub == sub).squeeze()
        action = self.sub_actions[sub_i][action]
        # return action
        if self.inspect_act(sub, action, topo_vect):
            return action
        else:  # return do nothing action.
            return self.action_space()

class RAMAActionConverter(SimpleDiscActionConverter):
    def __init__(self, env, mask, mask_hi, num_clusters, cluster_method, rule="c"):
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.node_num = len(env.action_space.sub_info)
        self.obs = env.get_obs()
        super().__init__(env, mask, mask_hi, rule)
        self.n = 0  # not relevant for MA
        self.current_ra = None
          
    def init_action_converter(self):
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        self.masked_sorted_sub = self.subs[sort_subs]
        
        # Perform clustering
        self.clusters = perform_clustering(self.cluster_method, self.node_num, self.masked_sorted_sub, self.num_clusters, self.obs)
        print(f"Clusters (before masking): {self.clusters}")
        self.ra_idx = [i for i in range(len(self.clusters))] # Equivalent of self.masked_sorted_sub for RA in terms of usage.
        self.cluster_actions = []
        self.n_cluster_actions = []

        self.act_to_sub = {} # action to substation mapping

        for cluster in self.clusters:
            cluster_action_ids = []
            for sub in cluster:
                if sub in self.masked_sorted_sub:
                    sub_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub)
                    for action in sub_actions:
                        self.act_to_sub[str(action)] = sub
                        cluster_action_ids.append(action)  # Actions are identifiable and unique
            self.cluster_actions.append(cluster_action_ids)
            self.n_cluster_actions.append(len(cluster_action_ids))

    def plan_act(self, goal, topo_vect, **kwargs):
        ra_idx, action = goal
        action = self.cluster_actions[ra_idx][action]
        # return action
        if self.inspect_act(ra_idx, action, topo_vect):
            self.current_ra = ra_idx
            return action
        else:  # return do nothing action.
            return self.action_space()

    def inspect_act(self, ra_idx, action, topo):
        # check if action is relevant or redundant for the entire cluster
        relevant = False
        sub_id = self.act_to_sub[str(action)]
        beg = self.sub_to_topo_begin[sub_id]
        end = self.sub_to_topo_end[sub_id]
        topo = topo[beg:end]
        new_topo = action.set_bus[beg:end]
    
        if np.any(topo != new_topo):
            return True
        return False


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
