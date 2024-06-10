import grid2op
print("local envs:", grid2op.list_available_local_env())
print("remote envs:", grid2op.list_available_remote_env())
print("test envs:", grid2op.list_available_test_env())
from grid2op.PlotGrid import PlotMatplot
import matplotlib.pyplot as plt

for env in grid2op.list_available_local_env():
    print(env)
    env = grid2op.make(env)
    plot_helper = PlotMatplot(env.observation_space)
    fig = plot_helper.plot_layout()
    plt.show()

for env in ['l2rpn_2019', 'l2rpn_case14_sandbox', 'l2rpn_icaps_2021_large', 'l2rpn_icaps_2021_small', 'l2rpn_idf_2023', 'l2rpn_neurips_2020_track1_large', 'l2rpn_neurips_2020_track1_small', 'l2rpn_neurips_2020_track2_large', 'l2rpn_neurips_2020_track2_small', 'l2rpn_wcci_2020', 'l2rpn_wcci_2022', 'rte_case14_realistic', 'rte_case14_redisp', 'wcci_test']:
    print(env)
    env = grid2op.make(env)
    plot_helper = PlotMatplot(env.observation_space)
    fig = plot_helper.plot_layout()
    plt.show()