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
        if seq[idx] > val and seq[idx] < min:
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
    min_obs = max(0, min_lt(obs, 0))
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    if norm == 0:
        norm = 1.
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


class CustomPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        return ((sum([space.shape[0] for space in obs_space[:2]]) + obs_space[2].shape[0]*obs_space[2].shape[1])*2,)

    def transform(self, observation):
        # if len(observation) == 111:
        return np.concatenate([norm_obs_clip(observation[0][0]), observation[0][1], observation[0][2].flatten(), norm_obs_clip(observation[1]), observation[2], observation[3].flatten()])
        one_hot = observation[-3:]
        return np.append(obs, one_hot)
        # else:
        #     return observation


class ConvModelPreprocessor(Preprocessor):
    def _init_shape(self, obs_space, options):
        out_shape = (obs_space[0].shape[0], obs_space[0].shape[1], sum([space.shape[2] for space in obs_space]))
        return out_shape

    def transform(self, observation):
        return np.concatenate([observation[0],
                               observation[1],
                               observation[2]], axis=2)



# class NoPreprocessor:
#     def _init_shape(self, obs_space, options):
#         num_features = 0
#         for space in obs_space:

