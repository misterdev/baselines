import time

import numpy as np
import torch
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from torch_training.dueling_double_dqn import Agent
from scoring.utils.misc_utils import run_test
from utils.observation_utils import normalize_observation

with open('parameters.txt', 'r') as inf:
    parameters = eval(inf.read())

# Parameter initialization
features_per_node = 9
tree_depth = 3
nodes = 0
for i in range(tree_depth + 1):
    nodes += np.power(4, i)
state_size = features_per_node * nodes
action_size = 5
action_dict = dict()
nr_trials_per_test = 100
test_results = []
test_times = []
test_dones = []
sequential_agent_test = False

# Load your agent
agent = Agent(state_size, action_size)
agent.qnetwork_local.load_state_dict(torch.load('../torch_training/Nets/avoid_checkpoint500.pth'))

# Load the necessary Observation Builder and Predictor
predictor = ShortestPathPredictorForRailEnv()
observation_builder = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)

start_time_scoring = time.time()

score_board = []
for test_nr in parameters:
    current_parameters = parameters[test_nr]
    test_score, test_dones, test_time = run_test(current_parameters, agent, observation_builder=observation_builder,
                                                 observation_wrapper=normalize_observation,
                                                 test_nr=test_nr, nr_trials_per_test=10)
    print('{} score was {:.3f} with {:.2f}% environments solved. Test took {:.2f} Seconds to complete.\n'.format(
        test_nr,
        np.mean(test_score), np.mean(test_dones) * 100, test_time))

    score_board.append([np.mean(test_score), np.mean(test_dones) * 100, test_time])
print('---------')
print(' RESULTS')
print('---------')
test_idx = 0
for test_nr in parameters:
    print('{} score was {:.3f}\twith {:.2f}% environments solved.\tTest took {:.2f} Seconds to complete.'.format(
        test_nr, score_board[test_idx][0], score_board[test_idx][1], score_board[test_idx][2]))
    test_idx += 1
