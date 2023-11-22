import numpy as np
from grid2op.Reward.baseReward import BaseReward
from grid2op.Reward import L2RPNReward
from grid2op.dtypes import dt_float


class LossReward(BaseReward):
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=None)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
            if is_illegal or is_ambiguous:
                return self.reward_illegal
            elif is_done:
                return self.reward_min
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10.0 - 9.0) * 0.1  # avg ~ 0.01
        return reward


class MarginReward(L2RPNReward):
    """
    This reward was used in the paper "Hierarchical Reinforcement Learning for Power Network Topology Control"

    Scaled version of L2RPNReward such that the reward falls between 0 and 1.
    Additionally -0.5 is awarded for illegal actions.
    """

    def initialize(self, env):
        self.reward_min = -0.5
        self.reward_illegal = -0.5
        self.reward_max = 1.0
        self.num_lines = env.backend.n_line

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap) / self.num_lines
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        x = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(dt_float(1.0) - x**2, np.zeros(x.shape, dtype=dt_float))
        return lines_capacity_usage_score
