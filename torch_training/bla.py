import getopt
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from importlib_resources import path

import torch_training.Nets
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import norm_obs_clip, split_tree

print("multi_agent_trainging.py (1)")

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)
    print("main1")
    random.seed(1)
    np.random.seed(1)

    """
    env = RailEnv(width=10,
                  height=20, obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()))
    env.load("./railway/complex_scene.pkl")
    file_load = True
    """

    x_dim = np.random.randint(8, 20)
    y_dim = np.random.randint(8, 20)
    n_agents = np.random.randint(3, 8)
    n_goals = n_agents + np.random.randint(0, 3)
    min_dist = int(0.75 * min(x_dim, y_dim))
    print("main2")

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=0),
                  obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()),
                  number_of_agents=n_agents)
    env.reset(True, True)
    file_load = False
    observation_helper = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
    env_renderer = RenderTool(env, gl="PILSVG", )
    handle = env.get_agent_handles()
    features_per_node = 9
    state_size = features_per_node * 85 * 2
    action_size = 5

    print("main3")

    # We set the number of episodes we would like to train on
    if 'n_trials' not in locals():
        n_trials = 30000
    max_steps = int(3 * (env.height + env.width))
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.9995
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    time_obs = deque(maxlen=2)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    agent = Agent(state_size, action_size, "FC", 0)

print("multi_agent_trainging.py (2)")

if __name__ == '__main__':
    print("main")
    main(sys.argv[1:])

print("multi_agent_trainging.py (3)")