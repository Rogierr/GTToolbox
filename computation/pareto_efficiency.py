import numpy as np

# Below is a pareto efficient algorithm retrieved from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python


def is_pareto_efficient_simple(rewards):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(rewards.shape[0], dtype = bool)
    for i, c in enumerate(rewards):
        print(i,c)
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(rewards[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def is_pareto_efficient(rewards, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(rewards.shape[0])
    n_points = rewards.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(rewards):
        nondominated_point_mask = np.any(rewards>rewards[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        rewards = rewards[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient