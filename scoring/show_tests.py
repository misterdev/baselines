import time

import numpy as np

from utils.misc_utils import render_test

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
test_idx = 0

for test_nr in parameters:
    current_parameters = parameters[test_nr]
    render_test(current_parameters, test_nr, nr_examples=2)

