import numpy as np
from ray.rllib.models.preprocessors import Preprocessor
from utils.observation_utils import norm_obs_clip

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

