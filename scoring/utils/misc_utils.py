import random
import time
from collections import deque

import numpy as np
from flatland.envs.generators import complex_rail_generator, rail_from_file
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_env import RailEnv

from utils.observation_utils import norm_obs_clip, split_tree, max_lt

# Time factor to test the max time allowed for an env.
max_time_factor = 1

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '_' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=" ")
    # Print New Line on Complete
    if iteration == total:
        print('')


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, eps=0):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice(np.arange(self.action_size))

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


def run_test(parameters, agent, test_nr=0, tree_depth=3):
    # Parameter initialization
    features_per_node = 9
    start_time_scoring = time.time()
    action_dict = dict()
    nr_trials_per_test = 100
    print('Running {} with (x_dim,y_dim) = ({},{}) and {} Agents.'.format(test_nr, parameters[0], parameters[1],
                                                                          parameters[2]))

    # Reset all measurements
    test_scores = []
    test_dones = []

    # Reset environment
    random.seed(parameters[3])
    np.random.seed(parameters[3])


    printProgressBar(0, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    for trial in range(nr_trials_per_test):
        # Reset the env
        file_name = "./Envs/{}/Level_{}.pkl".format(test_nr, trial)

        env = RailEnv(width=3,
                      height=3,
                      rail_generator=rail_from_file(file_name),
                      obs_builder_object=TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv()),
                      number_of_agents=1,
                      )

        obs = env.reset()
        agent_obs = [None] * env.get_num_agents()
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=9,
                                                    current_depth=0)
            data = norm_obs_clip(data, fixed_radius=10)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))


        # Run episode
        trial_score = 0
        max_steps = int(max_time_factor*  (env.height + env.width))
        for step in range(max_steps):

            for a in range(env.get_num_agents()):
                action = agent.act(agent_obs[a], eps=0)
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                        num_features_per_node=features_per_node,
                                                        current_depth=0)
                data = norm_obs_clip(data, fixed_radius=10)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                agent_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
                trial_score += np.mean(all_rewards[a])
            if done['__all__']:
                break
        test_scores.append(trial_score / max_steps)
        test_dones.append(done['__all__'])
        printProgressBar(trial + 1, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    end_time_scoring = time.time()
    tot_test_time = end_time_scoring - start_time_scoring
    return test_scores, test_dones, tot_test_time


def create_testsfiles(parameters, test_nr=0):
    # Parameter initialization
    nr_trials_per_test = 100
    print('Creating {} with (x_dim,y_dim) = ({},{}) and {} Agents.'.format(test_nr, parameters[0], parameters[1],
                                                                           parameters[2]))
    # Reset environment
    random.seed(parameters[3])
    np.random.seed(parameters[3])
    nr_paths = max(4, parameters[2] + int(0.5 * parameters[2]))
    min_dist = int(min([parameters[0], parameters[1]]) * 0.75)
    env = RailEnv(width=parameters[0],
                  height=parameters[1],
                  rail_generator=complex_rail_generator(nr_start_goal=nr_paths, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=parameters[3]),
                  obs_builder_object=TreeObsForRailEnv(max_depth=2),
                  number_of_agents=parameters[2])
    printProgressBar(0, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    for trial in range(nr_trials_per_test):
        # Reset the env
        env.reset(True, True)
        env.save("./Envs/{}/Level_{}.pkl".format(test_nr, trial))
        printProgressBar(trial + 1, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)

    return


def render_test(parameters, test_nr=0, nr_examples=5):
    for trial in range(nr_examples):
        # Reset the env
        print('Showing {} Level {} with (x_dim,y_dim) = ({},{}) and {} Agents.'.format(test_nr, trial, parameters[0],
                                                                                       parameters[1],
                                                                                       parameters[2]))
        file_name = "./Envs/{}/Level_{}.pkl".format(test_nr, trial)

        env = RailEnv(width=1,
                      height=1,
                      rail_generator=rail_from_file(file_name),
                      obs_builder_object=TreeObsForRailEnv(max_depth=2),
                      number_of_agents=1,
                      )
        env_renderer = RenderTool(env, gl="PILSVG", )
        env_renderer.set_new_rail()

        env.reset(False, False)
        env_renderer.render_env(show=True, show_observations=False)

        time.sleep(0.1)
        env_renderer.close_window()
    return

def run_test_sequential(parameters, agent, test_nr=0, tree_depth=3):
    # Parameter initialization
    features_per_node = 9
    start_time_scoring = time.time()
    action_dict = dict()
    nr_trials_per_test = 100
    print('Running {} with (x_dim,y_dim) = ({},{}) and {} Agents.'.format(test_nr, parameters[0], parameters[1],
                                                                          parameters[2]))

    # Reset all measurements
    test_scores = []
    test_dones = []

    # Reset environment
    random.seed(parameters[3])
    np.random.seed(parameters[3])


    printProgressBar(0, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    for trial in range(nr_trials_per_test):
        # Reset the env
        file_name = "./Envs/{}/Level_{}.pkl".format(test_nr, trial)

        env = RailEnv(width=3,
                      height=3,
                      rail_generator=rail_from_file(file_name),
                      obs_builder_object=TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv()),
                      number_of_agents=1,
                      )

        obs = env.reset()
        done = env.dones
        # Run episode
        trial_score = 0
        max_steps = int(max_time_factor* (env.height + env.width))
        for step in range(max_steps):

            # Action
            acting_agent = 0
            for a in range(env.get_num_agents()):
                if done[a]:
                    acting_agent += 1
                if acting_agent == a:
                    action = agent.act(obs[acting_agent], eps=0)
                else:
                    action = 0
                action_dict.update({a: action})

            # Environment step

            obs, all_rewards, done, _ = env.step(action_dict)
            for a in range(env.get_num_agents()):
                trial_score += np.mean(all_rewards[a])
            if done['__all__']:
                break
        test_scores.append(trial_score / max_steps)
        test_dones.append(done['__all__'])
        printProgressBar(trial + 1, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    end_time_scoring = time.time()
    tot_test_time = end_time_scoring - start_time_scoring
    return test_scores, test_dones, tot_test_time
