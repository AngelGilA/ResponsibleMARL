import itertools
import numpy as np


def enumerated_product(n_items, items):
    yield from zip(itertools.product(*(range(x) for x in n_items)), itertools.product(*items))


class SimpleDiscActionConverter:
    def __init__(self, env, mask, mask_hi, rule='c', **kwargs):
        self.action_space = env.action_space
        self.mask = mask
        self.mask_hi = mask_hi
        self.rule = rule
        self.sub_mask = []  # mask for parsing actionable topology
        self.psubs = []  # actionable substation IDs
        self.masked_sub_begin = []
        self.masked_sub_end = []
        self.init_sub_topo()
        self.init_action_converter()


    def init_sub_topo(self):
        # mask_hi to be included later
        self.subs = np.flatnonzero(self.action_space.sub_info > self.mask)
        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def init_action_converter(self):
        # sort subs descending:
        sort_subs = np.argsort(-self.action_space.sub_info[self.action_space.sub_info > self.mask])
        self.masked_sorted_sub = self.subs[sort_subs]

        self.actions = []
        self.n_sub_actions = np.zeros(len(self.masked_sorted_sub), dtype=int)
        for i, sub in enumerate(self.masked_sorted_sub):
            topo_actions = self.action_space.get_all_unitary_topologies_set(self.action_space, sub)
            self.actions += topo_actions
            self.n_sub_actions[i] = len(topo_actions)
            self.sub_mask.extend(range(self.sub_to_topo_begin[sub], self.sub_to_topo_end[sub]))
        self.sub_pos = self.n_sub_actions.cumsum()
        self.n = sum(self.n_sub_actions)

    # def get_action(self, action_probs, sample):
    #     action_probs = action_probs.squeeze(0)
    #     if sample:
    #         return Categorical(action_probs).sample().cpu()
    #     else:
    #         return action_probs.argmax()
    #     # TODO: use top N actions with prob

    def plan_act(self, action, topo_vect, **kwargs):
        idx = np.argmin(self.sub_pos < int(action))
        sub_id = self.masked_sorted_sub[idx]
        action = self.actions[action]
        if self.inspect_act(sub_id, action, topo_vect):
            return action
        else: # return do nothing action.
            return self.action_space()

    def inspect_act(self, sub_id, action, topo):
        # check if action is relevant or redundant
        beg = self.sub_to_topo_begin[sub_id]
        end = self.sub_to_topo_end[sub_id]
        topo = topo[beg:end]
        new_topo = action.set_bus[beg:end]
        if np.any(topo != new_topo):
            return True
        return False

    def optimize_plan(self, obs, plan):
        if len(plan):
            # remove action in case of cooldown
            cooldown_list = obs.time_before_cooldown_sub
            cooldown_list = np.array([cooldown_list[i[0]] for i in plan])
            if np.min(cooldown_list) > 0:
                plan = []
        return plan

    def convert_act(self, sub_id, action):
        return action


class DiscActionConverter(SimpleDiscActionConverter):
    def __init__(self, env, mask, mask_hi, rule='c'):
        super().__init__(env, mask, mask_hi, rule)
        self.actions = self.preprocessing(env.env_name)
        self.n = len(self.actions)

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
            self.sub_line_or.append(np.flatnonzero(self.action_space.line_or_to_subid==sub))
            self.sub_line_ex.append(np.flatnonzero(self.action_space.line_ex_to_subid == sub))

    def preprocessing(self, env_name):
        import grid2op
        from grid2op.Parameters import Parameters
        from lightsim2grid import LightSimBackend
        from datetime import datetime
        print("Start preprocessing...")
        tic = datetime.now()
        # Create parameters
        p = Parameters()
        # Disable lines disconnections due to overflows
        p.NO_OVERFLOW_DISCONNECTION = True
        # Give Parameters instance to make, so its used
        temp_env = grid2op.make(env_name, param=p, test=True, backend=LightSimBackend())
        actions = []
        for idx, action_combo in enumerated_product(self.n_sub_actions, self.sub_actions):
            obs = temp_env.reset()
            for action in action_combo:
                obs, _, done,_ = temp_env.step(action)
                if done:
                    break
            if not done:
                actions.append(idx)

        toc = datetime.now()
        print("Duration of preprocessing: ", toc - tic)
        return actions

    def plan_act(self, goal, topo_vect, **kwargs):
        plan = []
        # goal is currently the idx of the goal in self.actions.
        goal = self.actions[goal]
        for i, sub_id in enumerate(self.masked_sorted_sub):
            action = self.sub_actions[i][goal[i]]
            if self.inspect_act(sub_id, action, topo_vect):
                plan.append((sub_id, action))
        return plan

    def optimize_plan(self, obs, plan):
        # TO DO: design rules to optimize the plan
        if len(plan):
            if (self.rule == 'c') & (len(plan) > 1):
                # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
                # which applies an action to substations that require urgent care.
                plan = self.heuristic_order(obs, plan)

            # sort by cooldown_sub
            cooldown_list = obs.time_before_cooldown_sub
            cooldown_list = np.array([cooldown_list[i[0]] for i in plan])
            if np.min(cooldown_list) > 0:
                plan = []
            elif self.rule != 'o':
                plan = [i[0] for i in sorted(list(zip(plan, cooldown_list)), key=lambda x: -x[1])]
        return plan

    def heuristic_order(self, obs, plan):
        # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
        # which applies an action to substations that require urgent care.
        rhos = []
        for sub_plan in plan:
            sub_i = np.flatnonzero(self.masked_sorted_sub == sub_plan[0]).squeeze()
            rho = np.append(obs.rho[self.sub_line_or[sub_i]].copy(), obs.rho[self.sub_line_ex[sub_i]].copy())
            rho[rho == 0] = 3
            rho_max = rho.max()
            rho_mean = rho.mean()
            rhos.append((rho_max, rho_mean))
        order = sorted(zip(plan, rhos), key=lambda x: (-x[1][0], -x[1][1]))
        return list(list(zip(*order))[0])