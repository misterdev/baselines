import numpy as np
from ray.rllib.models.preprocessors import Preprocessor

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


class TreeObsPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        print(options)
        self.step_memory = options["custom_options"]["step_memory"]
        return sum([space.shape[0] for space in obs_space]),

    def transform(self, observation):

        if self.step_memory == 2:
            data = norm_obs_clip(observation[0][0])
            distance = norm_obs_clip(observation[0][1])
            agent_data = np.clip(observation[0][2], -1, 1)
            data2 = norm_obs_clip(observation[1][0])
            distance2 = norm_obs_clip(observation[1][1])
            agent_data2 = np.clip(observation[1][2], -1, 1)
        else:
            data = norm_obs_clip(observation[0])
            distance = norm_obs_clip(observation[1])
            agent_data = np.clip(observation[2], -1, 1)

        return np.concatenate((np.concatenate((np.concatenate((data, distance)), agent_data)), np.concatenate((np.concatenate((data2, distance2)), agent_data2))))
