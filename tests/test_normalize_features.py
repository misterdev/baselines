import random

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator

from utils.observation_utils import normalize_observation


def test_normalize_features():

    random.seed(1)
    np.random.seed(1)
    max_depth = 4

    for i in range(10):
        tree_observer = TreeObsForRailEnv(max_depth=max_depth)
        next_rand_number = random.randint(0, 100)

        env = RailEnv(width=10,
                      height=10,
                      rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
                                                            seed=next_rand_number),
                      schedule_generator=complex_schedule_generator(),
                      number_of_agents=1,
                      obs_builder_object=tree_observer)

        obs, all_rewards, done, _ = env.step({0: 0})

        obs_new = tree_observer.get()
        # data, distance, agent_data = split_tree(tree=np.array(obs_old), num_features_per_node=11)
        data_normalized = normalize_observation(obs_new, max_depth, observation_radius=10)

        filename = 'testdata/test_array_{}.csv'.format(i)
        data_loaded = np.loadtxt(filename, delimiter=',')

        assert np.allclose(data_loaded, data_normalized)

