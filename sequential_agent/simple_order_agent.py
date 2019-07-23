import numpy as np
from utils.observation_utils import split_tree, min_lt


class OrderedAgent:

    def __init__(self):
        self.action_size = 5

    def act(self, state, eps=0):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        _, distance, _ = split_tree(tree=np.array(state), num_features_per_node=9,
                                    current_depth=0)
        distance = distance[1:]
        min_dist = min_lt(distance, 0)
        min_direction = np.where(distance == min_dist)
        if len(min_direction[0]) > 1:
            return min_direction[0][0] + 1
        return min_direction[0] + 1

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return
