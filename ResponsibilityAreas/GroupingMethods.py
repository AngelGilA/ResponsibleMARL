
# import random

# def dummy_cluster(case, masked_subs, n_clusters, network): # [5 1 3 4 8 2 12]
#     if case == 5:
#         return [[0, 1, 2], [3, 4]]  # Hardcoded clusters for 5 substations
#     elif case == 14:
#         if n_clusters == 2:
#             return [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13]]
#         elif n_clusters == 3:
#             return [[0, 1, 2, 3, 4],[5, 6, 7, 8, 9],[10, 11, 12, 13]]
#         elif n_clusters == 4:
#             return [[0, 1, 4],[2, 3, 6, 7],[5, 11, 12],[8, 9, 10, 13]]
#         else:
#             return [[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
#         # Generate 5 random clusters for 14 substations
#         all_substations = list(range(14))  # Assuming 14 substations are numbered 0 through 13
#         random.shuffle(all_substations)
#         clusters = [all_substations[i::5] for i in range(5)]  # Divide into 5 clusters
#         return clusters
#     else:
#         raise ValueError(f"Unsupported case: {case}")
    


# import random
# import numpy as np
# from sklearn.cluster import KMeans
# from collections import deque, defaultdict
# import os
# os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Or the number of cores you want to use
# os.environ["OMP_NUM_THREADS"] = "1"


# # Function to create the graph representation from network data
# def create_graph(network):
#     graph = defaultdict(list)
#     for line_or, line_ex in zip(network.line_or_to_subid, network.line_ex_to_subid):
#         graph[line_or].append(line_ex)
#         graph[line_ex].append(line_or)
#     print(f"graph:",graph)
#     return graph

# # BFS function to compute shortest paths
# def bfs(graph, start):
#     distances = {node: float('inf') for node in graph}
#     distances[start] = 0
#     queue = deque([start])
    
#     while queue:
#         current = queue.popleft()
#         for neighbor in graph[current]:
#             if distances[neighbor] == float('inf'):
#                 distances[neighbor] = distances[current] + 1
#                 queue.append(neighbor)
#     print(f"distances:",distances)
#     return distances

# # Function to compute the distance matrix using BFS
# def compute_distance_matrix(graph, num_substations):
#     distance_matrix = np.full((num_substations, num_substations), float('inf'))
#     for sub in range(num_substations):
#         distances = bfs(graph, sub)
#         for target in distances:
#             distance_matrix[sub][target] = distances[target]
#     print(f"distance_matrix:",distance_matrix)
#     return distance_matrix

# # Clustering function
# def cluster_substations(distance_matrix, num_clusters):
#     clustering_model = KMeans(n_clusters=num_clusters)
#     clusters = clustering_model.fit_predict(distance_matrix)
#     cluster_dict = defaultdict(list)
#     for substation, cluster_id in enumerate(clusters):
#         cluster_dict[cluster_id].append(substation)
#     print(f"cluster_dict:",cluster_dict)
#     return list(cluster_dict.values())

# # Updated dummy_cluster function
# def dummy_cluster(case, masked_subs, n_clusters, network):
#     graph = create_graph(network)
#     num_substations = len(network.grid_layout)
#     distance_matrix = compute_distance_matrix(graph, num_substations)

#     if case == 5:
#         return [[0, 1, 2], [3, 4]]  # Hardcoded clusters for 5 substations
#     elif case == 14:
#         return cluster_substations(distance_matrix, n_clusters)
#     else:
#         raise ValueError(f"Unsupported case: {case}")

# import grid2op
# import matplotlib.pyplot as plt
# import numpy as np

# env = grid2op.make("rte_case14_realistic", test=True)
# obs = env.get_obs()
# masked_subs = np.array([5, 1, 3, 4, 8, 2, 12])
# n_clusters = 2
# clusters = dummy_cluster(14, masked_subs, n_clusters, obs)

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Function to create the graph representation from network data
def create_graph(network):
    graph = nx.Graph()
    for line_or, line_ex in zip(network.line_or_to_subid, network.line_ex_to_subid):
        graph.add_edge(line_or, line_ex)
    return graph

# Function to compute the distance matrix using NetworkX
def compute_distance_matrix(graph, num_substations):
    distance_matrix = np.full((num_substations, num_substations), float('inf'))
    for sub in range(num_substations):
        lengths = nx.single_source_shortest_path_length(graph, sub)
        for target, length in lengths.items():
            distance_matrix[sub][target] = length
    return distance_matrix

# KMeans Clustering Function
def kmeans_basic(distance_matrix, num_clusters):
    clustering_model = KMeans(n_clusters=num_clusters)
    clusters = clustering_model.fit_predict(distance_matrix)
    cluster_dict = defaultdict(list)
    for substation, cluster_id in enumerate(clusters):
        cluster_dict[cluster_id].append(substation)
    return list(cluster_dict.values())

# NetworkX Kernighan-Lin Clustering Function
def nx_kernighan_lin_clustering(graph, num_clusters):
    nodes = list(graph.nodes)
    partitions = [nodes]
    
    # Iteratively partition the graph
    while len(partitions) < num_clusters:
        new_partitions = []
        for partition in partitions:
            if len(partition) > 1:
                subgraph = graph.subgraph(partition)
                part1, part2 = nx.algorithms.community.kernighan_lin_bisection(subgraph)
                new_partitions.extend([list(part1), list(part2)])
            else:
                new_partitions.append(partition)
        partitions = new_partitions

    # Organize the nodes into clusters
    cluster_dict = defaultdict(list)
    for idx, partition in enumerate(partitions):
        for node in partition:
            cluster_dict[idx].append(node)

    return list(cluster_dict.values())

# Recursive Bisection Function
def recursive_bisection(graph, num_clusters, max_iterations=10):
    def balance_bisection(subgraph):
        best_cut = None
        best_balance = float('inf')
        for _ in range(max_iterations):
            part1, part2 = nx.algorithms.community.kernighan_lin_bisection(subgraph)
            balance = abs(len(part1) - len(part2))
            if balance < best_balance:
                best_balance = balance
                best_cut = (part1, part2)
            if best_balance == 0:
                break
        return best_cut

    nodes = list(graph.nodes)
    partitions = [nodes]

    while len(partitions) < num_clusters:
        new_partitions = []
        for partition in partitions:
            if len(partition) > 1:
                subgraph = graph.subgraph(partition)
                part1, part2 = balance_bisection(subgraph)
                new_partitions.extend([list(part1), list(part2)])
            else:
                new_partitions.append(partition)
        partitions = new_partitions

    if len(partitions) > num_clusters:
        # Merge the smallest partitions to match the desired number of clusters
        while len(partitions) > num_clusters:
            partitions.sort(key=len)
            part1 = partitions.pop(0)
            part2 = partitions.pop(0)
            merged_partition = list(part1) + list(part2)
            partitions.append(merged_partition)

    # Organize the nodes into clusters
    cluster_dict = defaultdict(list)
    for idx, partition in enumerate(partitions):
        for node in partition:
            cluster_dict[idx].append(node)

    return list(cluster_dict.values())

# Function to perform clustering based on selected method
def perform_clustering(method, case, masked_subs, n_clusters, network):
    graph = create_graph(network)
    num_substations = len(network.grid_layout)
    distance_matrix = compute_distance_matrix(graph, num_substations)

    if case == 5:
        return [[0, 1, 2], [3, 4]]  # Hardcoded clusters for 5 substations
    elif case == 14:
        if method == 'kmeans':
            return kmeans_basic(distance_matrix, n_clusters)
        elif method == 'nx_kernighan_lin':
            return nx_kernighan_lin_clustering(graph, n_clusters)
        elif method == 'recursive_bisection':
            return recursive_bisection(graph, n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    else:
        raise ValueError(f"Unsupported case: {case}")

# # Usage example
# import grid2op

# # Create the environment and observation
# env = grid2op.make("rte_case14_realistic", test=True)
# obs = env.get_obs()
# masked_subs = np.array([5, 1, 3, 4, 8, 2, 12])
# n_clusters = 2

# # Choose clustering method: 'kmeans' or 'nx_kernighan_lin'
# method = 'kmeans'

# # Run the perform_clustering function with the selected method
# clusters = perform_clustering(method, 14, masked_subs, n_clusters, obs)
# print(f"clusters (KMeans): {clusters}")

# # Switch to Recursive Bisection and run again
# method = 'recursive_bisection'
# clusters = perform_clustering(method, 14, masked_subs, n_clusters, obs)
# print(f"clusters (Recursive Bisection): {clusters}")

# # Switch to NetworkX Kernighan-Lin and run again
# method = 'nx_kernighan_lin'
# clusters = perform_clustering(method, 14, masked_subs, n_clusters, obs)
# print(f"clusters (Kernighan-Lin): {clusters}")
