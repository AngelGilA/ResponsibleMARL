
import random

def dummy_cluster(case):
    print(case)
    if case == 5:
        return [[0, 1, 2], [3, 4]]  # Hardcoded clusters for 5 substations
    elif case == 14:
        return [[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
        return [[0, 1, 4],[2, 3, 6, 7],[5, 11, 12],[8, 9, 10, 13]]
        # Generate 5 random clusters for 14 substations
        all_substations = list(range(14))  # Assuming 14 substations are numbered 0 through 13
        random.shuffle(all_substations)
        clusters = [all_substations[i::5] for i in range(5)]  # Divide into 5 clusters
        return clusters
    else:
        raise ValueError(f"Unsupported case: {case}")