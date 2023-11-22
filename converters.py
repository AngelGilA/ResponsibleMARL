import torch
import itertools
import numpy as np


def enumerated_product(n_items, items):
    yield from zip(itertools.product(*(range(x) for x in n_items)), itertools.product(*items))


class SimpleDiscActionConverter:
    def __init__(self, env, mask, mask_hi, rule="c", **kwargs):
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

    def plan_act(self, action, topo_vect, **kwargs):
        idx = np.argmin(self.sub_pos < int(action))
        sub_id = self.masked_sorted_sub[idx]
        action = self.actions[action]
        if self.inspect_act(sub_id, action, topo_vect):
            return action
        else:  # return do nothing action.
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


class GoalActionConverter(SimpleDiscActionConverter):
    def __init__(self, env, mask, mask_hi, rule="c"):
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
            self.sub_line_or.append(np.flatnonzero(self.action_space.line_or_to_subid == sub))
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
                obs, _, done, _ = temp_env.step(action)
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
            if (self.rule == "c") & (len(plan) > 1):
                # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
                # which applies an action to substations that require urgent care.
                plan = self.heuristic_order(obs, plan)

            # sort by cooldown_sub
            cooldown_list = obs.time_before_cooldown_sub
            cooldown_list = np.array([cooldown_list[i[0]] for i in plan])
            if np.min(cooldown_list) > 0:
                plan = []
            elif self.rule != "o":
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


class ActionConverter(SimpleDiscActionConverter):
    """
    Action Converter as defined by SMAAC
    """

    def __init__(self, env, mask, mask_hi, rule="c", bus_thresh=0.1):
        super().__init__(env, mask, mask_hi, rule)
        self.bus_thr = bus_thresh

    def init_sub_topo(self):
        # parse substation info
        self.subs = [{"e": [], "o": [], "g": [], "l": []} for _ in range(self.action_space.n_sub)]
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid):
            self.subs[sub_id]["g"].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid):
            self.subs[sub_id]["l"].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid):
            self.subs[sub_id]["o"].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid):
            self.subs[sub_id]["e"].append(ex_id)

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def init_action_converter(self):
        idx = 0
        for i, num_topo in enumerate(self.action_space.sub_info):
            if num_topo > self.mask and num_topo < self.mask_hi:
                self.sub_mask.extend(range(self.sub_to_topo_begin[i] + 1, self.sub_to_topo_end[i]))
                self.psubs.append(i)
                self.masked_sub_begin.append(idx)
                idx += num_topo - 1
                self.masked_sub_end.append(idx)

            else:
                self.masked_sub_begin.append(-1)
                self.masked_sub_end.append(-1)
        self.n = len(self.sub_mask)

        if self.rule == "f":
            # FIXED gives priority to substations randomly that are predefined and fixed during training.
            # We implement this low-level agent to find out whether our high- level agent can manage the
            # power network on the poor low- level agent. (2)
            if self.action_space.n_sub == 5:
                self.masked_sorted_sub = [4, 0, 1, 3, 2]
            elif self.action_space.n_sub == 14:
                self.masked_sorted_sub = [13, 5, 0, 12, 9, 6, 10, 1, 11, 3, 4, 7, 2]
            elif self.action_space.n_sub == 36:  # mask = 5
                self.masked_sorted_sub = [9, 33, 29, 7, 21, 1, 4, 23, 16, 26, 35]
                if self.mask == 4:
                    self.masked_sorted_sub = [35, 23, 9, 33, 4, 28, 1, 32, 13, 21, 26, 29, 16, 22, 7, 27]
        else:
            # rule d: DESC imposes a priority on large substations, i.e. many connected elements.
            # A change to a large substation can be seen as making a large change in the overall
            # topology with a single action. (hard coded unfortunately...)
            if self.action_space.n_sub == 5:
                self.masked_sorted_sub = [0, 3, 2, 1, 4]
            elif self.action_space.n_sub == 14:
                self.masked_sorted_sub = [5, 1, 3, 4, 2, 12, 0, 11, 13, 10, 9, 6, 7]
            elif self.action_space.n_sub == 36:  # mask = 5
                self.masked_sorted_sub = [16, 23, 21, 26, 33, 29, 35, 9, 7, 4, 1]
                if self.mask == 4:
                    self.masked_sorted_sub += [22, 27, 28, 32, 13]

        # powerlines which are not controllable by bus assignment action
        self.lonely_lines = set()
        for i in range(self.action_space.n_line):
            if (self.action_space.line_or_to_subid[i] not in self.psubs) and (
                self.action_space.line_ex_to_subid[i] not in self.psubs
            ):
                self.lonely_lines.add(i)
        self.lonely_lines = list(self.lonely_lines)
        print("Lonely line", len(self.lonely_lines), self.lonely_lines)
        print(
            "Masked sorted topology",
            len(self.masked_sorted_sub),
            self.masked_sorted_sub,
        )

    def plan_act(self, goal, topo_vect, sub_order_score=None, **kwargs):
        bus_goal = np.where(goal.squeeze(0) > self.bus_thr, 2, 1)
        plan = []

        if sub_order_score is None:
            sub_order = self.masked_sorted_sub
        else:
            sub_order = [
                i[0] for i in sorted(list(zip(self.masked_sorted_sub, sub_order_score.tolist())), key=lambda x: -x[1])
            ]

        for sub_id in sub_order:
            beg = self.masked_sub_begin[sub_id]
            end = self.masked_sub_end[sub_id]
            new_topo = bus_goal[beg:end]
            topo = topo_vect[beg:end]
            new_topo, same = self.inspect_act(sub_id, new_topo, topo)
            if not same:
                # Assign sequentially actions from the goal
                plan.append((sub_id, new_topo))

        return plan

    def optimize_plan(self, obs, plan):
        if self.rule == "c":
            # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
            # which applies an action to substations that require urgent care.
            plan = self.heuristic_order(obs, plan)

        # sort by cooldown_sub
        if len(plan):
            cooldown_list = obs.time_before_cooldown_sub
            cooldown_list = np.array([cooldown_list[i[0]] for i in plan])
            if np.min(cooldown_list) > 0:
                plan = []
            elif self.rule != "o":
                plan = [i[0] for i in sorted(list(zip(plan, cooldown_list)), key=lambda x: -x[1])]

        return plan

    def inspect_act(self, sub_id, new_topo, topo):
        # Correct illegal action collect original ids
        exs = self.subs[sub_id]["e"]
        ors = self.subs[sub_id]["o"]
        lines = exs + ors  # [line_id0, line_id1, line_id2, ...]

        # minimal prevention of isolation
        line_idx = len(lines) - 1
        if (new_topo[:line_idx] == 1).all() * (new_topo[line_idx:] != 1).any():
            new_topo = np.ones_like(new_topo)

        # Compare obs.topo_vect and goal, then parse partial order from whole topological sort
        already_same = np.all(new_topo == topo)
        return new_topo, already_same

    def heuristic_order(self, obs, low_actions):
        # rule c: CAPA gives high priority to substations with lines under high utilization of their capacity,
        # which applies an action to substations that require urgent care.
        if len(low_actions) == 0:
            return []
        rhos = []
        for item in low_actions:
            sub_id = item[0]
            lines = self.subs[sub_id]["e"] + self.subs[sub_id]["o"]
            rho = obs.rho[lines].copy()
            rho[rho == 0] = 3
            rho_max = rho.max()
            rho_mean = rho.mean()
            rhos.append((rho_max, rho_mean))
        order = sorted(zip(low_actions, rhos), key=lambda x: (-x[1][0], -x[1][1]))
        return list(list(zip(*order))[0])

    def convert_act(self, sub_id, new_topo):
        new_topo = [1] + new_topo.tolist()
        act = self.action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
        return act


class ObsConverter:
    def __init__(self, env, danger, device, attr=["p_i", "p_l", "r", "o", "d", "m"]):
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.danger = danger
        self.device = device
        self.thermal_limit_under400 = torch.from_numpy(env._thermal_limit_a < 400)
        self.attr = attr
        if "p" in self.attr:  # make sure old versions still work properly
            self.attr.extend(["p_i", "p_l"])
        n_power_attr = len([i for i in self.attr if i.startswith("p")])
        self.n_feature = len(self.attr) - (n_power_attr > 1) * (n_power_attr - 1)
        self.init_obs_converter()

    def _get_attr_pos(self, list_attr):
        all_obs_ranges = []
        idx = self.obs_space.shape
        for attr in list_attr:
            pos = self.obs_space.attr_list_vect.index(attr)
            start = sum(idx[:pos])
            end = start + idx[pos]
            all_obs_ranges.append(np.arange(start, end))
        return all_obs_ranges

    def init_obs_converter(self):
        list_attr = [
            "gen_p",
            "load_p",
            "p_or",
            "p_ex",
            "rho",
            "timestep_overflow",
            "time_next_maintenance",
            "topo_vect",
        ]
        (
            self.pp,
            self.lp,
            self.op,
            self.ep,
            self.rho,
            self.over,
            self.main,
            self.topo,
        ) = self._get_attr_pos(list_attr)

        dim_topo = self.obs_space.dim_topo  # self.idx[-7]
        self.last_topo = np.ones(dim_topo, dtype=np.int32)

    def convert_obs(self, o):
        # o.shape : (B, O)
        # output (Batch, Node, Feature)
        length = self.obs_space.dim_topo  # N

        attr_list = []
        p = False
        p_ = torch.zeros(o.size(0), length)  # (B, N)
        if "p_i" in self.attr:
            # active power p
            p = True
            p_[..., self.obs_space.gen_pos_topo_vect] = o[..., self.pp]
            p_[..., self.obs_space.load_pos_topo_vect] = o[..., self.lp]
        if "p_l" in self.attr:
            # active power p
            p = True
            p_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.op]
            p_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.ep]
        if p:
            attr_list.append(p_)
        if "r" in self.attr:
            # rho (powerline usage ratio)
            rho_ = torch.zeros(o.size(0), length)
            rho_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.rho]
            rho_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.rho]
            attr_list.append(rho_)
        if "d" in self.attr:
            # whether each line is in danger
            danger_ = torch.zeros(o.size(0), length)
            danger = ((o[..., self.rho] >= self.danger - 0.05) & self.thermal_limit_under400) | (
                o[..., self.rho] >= self.danger
            )
            danger_[..., self.obs_space.line_or_pos_topo_vect] = danger.float()
            danger_[..., self.obs_space.line_ex_pos_topo_vect] = danger.float()
            attr_list.append(danger_)
        if "o" in self.attr:
            # whether overflow occurs in each powerline
            over_ = torch.zeros(o.size(0), length)
            over_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.over] / 3
            over_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.over] / 3
            attr_list.append(over_)
        if "m" in self.attr:
            # whether each powerline is in maintenance
            main_ = torch.zeros(o.size(0), length)
            temp = torch.zeros_like(o[..., self.main])
            temp[o[..., self.main] == 0] = 1
            main_[..., self.obs_space.line_or_pos_topo_vect] = temp
            main_[..., self.obs_space.line_ex_pos_topo_vect] = temp
            attr_list.append(main_)

        # current bus assignment
        topo_ = torch.clamp(o[..., self.topo] - 1, -1)

        state = torch.stack(attr_list, dim=2)  # B, N, F
        return state, topo_.unsqueeze(-1)

    def convert_fc(self, prod_p, load_p, mean, std):
        prod_p = prod_p.astype("float32")
        load_p = load_p.astype("float32")
        forecast_prod_p = (torch.from_numpy(prod_p) - mean[..., self.pp]) / std[..., self.pp]
        forecast_load_p = (torch.from_numpy(load_p) - mean[..., self.lp]) / std[..., self.lp]
        # active power p forecast at time t
        p_forcast_ = torch.zeros(1, self.obs_space.dim_topo)
        p_forcast_[..., self.obs_space.gen_pos_topo_vect] = forecast_prod_p
        p_forcast_[..., self.obs_space.load_pos_topo_vect] = forecast_load_p
        return p_forcast_.unsqueeze(-1)
