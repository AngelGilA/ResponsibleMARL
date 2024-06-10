import grid2op
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm  # for easy progress bar
display_tqdm = False  # this is set to False for ease with the unitt test, feel free to set it to True
from grid2op.PlotGrid import PlotMatplot
env = grid2op.make("rte_case14_realistic", test=True)
# rte_case5_example, 'l2rpn_2019', 'l2rpn_case14_sandbox', 'l2rpn_icaps_2021_large', 'l2rpn_icaps_2021_small', 'l2rpn_idf_2023', 'l2rpn_neurips_2020_track1_large', 'l2rpn_neurips_2020_track1_small', 'l2rpn_neurips_2020_track2_large', 'l2rpn_neurips_2020_track2_small', 'l2rpn_wcci_2020', 'l2rpn_wcci_2022', 'rte_case14_realistic', 'rte_case14_redisp', 'wcci_test'

plot_helper = PlotMatplot(env.observation_space)

from grid2op.Agent import DoNothingAgent
my_agent = DoNothingAgent(env.action_space)


all_obs = []
obs = env.reset()
"""
all_obs.append(obs)
reward = env.reward_range[0]
done = False
nb_step = 0
with tqdm(total=env.chronics_handler.max_timestep(), disable=not display_tqdm) as pbar:
    while True:
        action = my_agent.act(obs, reward, done)

        obs, reward, done, _ = env.step(action)
        pbar.update(1)
        if done:
            break
        all_obs.append(obs)
        nb_step += 1

print("Number of timesteps computed: {}".format(nb_step))
print("Total maximum number of timesteps possible: {}".format(env.chronics_handler.max_timestep()))

last_obs = all_obs[-1]
_ = plot_helper.plot_obs(last_obs)
plt.show()
"""

all_subs = np.flatnonzero(env.action_space.sub_info >= 0)
subs = np.flatnonzero(env.action_space.sub_info > 3)
print(all_subs, subs)

print(env.action_space.sub_info)

print(obs.topo_vect)


print(env.action_space.get_all_unitary_topologies_set(env.action_space, 0))