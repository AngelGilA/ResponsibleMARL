import grid2op
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm  # for easy progress bar
display_tqdm = False  # this is set to False for ease with the unitt test, feel free to set it to True
from grid2op.PlotGrid import PlotMatplot
env = grid2op.make("rte_case14_realistic", test=True)

obs = env.get_obs()

print("Loads:", obs.gen_margin_up, obs.gen_margin_down, obs.load_p, obs.load_q)
print("generation details:", obs.gen_p,obs.gen_pmax,obs.gen_pmin)
print("elements to substation:", obs.gen_to_subid, obs.load_to_subid, obs.line_or_to_subid, obs.line_ex_to_subid)
# Step 1: Call the function to get the connectivity matrix
connectivity_matrix = obs.connectivity_matrix()

# Step 2: Define the chunk you want to visualize. For example, let's say you want to see the first 10 rows and columns
start_row, end_row = 0, 10
start_col, end_col = 0, 10

# Step 3: Slice the matrix
chunk = connectivity_matrix[start_row:end_row, start_col:end_col]
print(connectivity_matrix.shape)
# Step 4: Print the sliced matrix
print("Chunk of the connectivity matrix:")
print(chunk)
print("Topologies:", obs.grid_layout)
a = obs.get_elements_graph
print("Total elements:",len(obs.topo_vect))
print(obs.load_pos_topo_vect, obs.gen_pos_topo_vect)
print(obs.n_gen, obs.n_line, obs.n_load, obs.n_sub)
print(obs.name_gen)
print(obs.shape)
print(obs.sub_info)
print(obs.sub_topology)
print(obs._aux_get_nb_ts)
print(obs._bus_connectivity_matrix_)
print(obs._check_sub_pos)
print(obs._compute_sub_pos)
print(obs._compute_pos_big_topo)