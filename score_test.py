import time

import numpy as np

from utils.misc_utils import RandomAgent, run_test

with open('parameters.txt','r') as inf:
    parameters = eval(inf.read())

# Parameter initialization
features_per_node = 9
tree_depth = 3
nodes = 0
for i in range(tree_depth + 1):
    nodes += np.power(4, i)
state_size = features_per_node * nodes * 2
action_size = 5
action_dict = dict()
nr_trials_per_test = 100
test_results = []
test_times = []
test_dones = []
agent = RandomAgent(state_size, action_size)
start_time_scoring = time.time()
test_idx = 0
score_board = []
for test_nr in parameters:
    current_parameters = parameters[test_nr]
    test_score, test_dones, test_time = run_test(current_parameters, agent, test_nr=test_idx)
    print('---------')
    print(' RESULTS')
    print('---------')
    print('{} score was {:.3f} with {:.2f}% environments solved. Test took {} Seconds to complete.\n\n\n'.format(
        test_nr,
        np.mean(test_score), np.mean(test_dones) * 100, test_time))
    test_idx += 1
    score_board.append([test_score, test_dones, test_times])
