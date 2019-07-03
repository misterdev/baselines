import numpy as np


def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_lt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    max_obs = max(1, max_lt(obs, 1000))
    min_obs = min(max_obs, min_lt(obs, 0))

    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    if norm == 0:
        norm = 1.
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def split_tree(tree, num_features_per_node=8, current_depth=0):
    """

    :param tree:
    :param num_features_per_node:
    :param prompt:
    :param current_depth:
    :return:
    """

    if len(tree) < num_features_per_node:
        return [], [], []

    depth = 0
    tmp = len(tree) / num_features_per_node - 1
    pow4 = 4
    while tmp > 0:
        tmp -= pow4
        depth += 1
        pow4 *= 4
    child_size = (len(tree) - num_features_per_node) // 4
    tree_data = tree[:4].tolist()
    distance_data = [tree[4]]
    agent_data = tree[5:num_features_per_node].tolist()
    for children in range(4):
        child_tree = tree[(num_features_per_node + children * child_size):
                          (num_features_per_node + (children + 1) * child_size)]
        tmp_tree_data, tmp_distance_data, tmp_agent_data = split_tree(child_tree,
                                                                      num_features_per_node,
                                                                      current_depth=current_depth + 1)
        if len(tmp_tree_data) > 0:
            tree_data.extend(tmp_tree_data)
            distance_data.extend(tmp_distance_data)
            agent_data.extend(tmp_agent_data)
    return tree_data, distance_data, agent_data
