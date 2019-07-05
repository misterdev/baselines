import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_training.dueling_double_dqn import Agent
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.generators import complex_rail_generator
from utils.observation_utils import norm_obs_clip, split_tree
from flatland.utils.rendertools import RenderTool
from utils.misc_utils import printProgressBar, RandomAgent


with open('parameters.txt','r') as inf:
    parameters = eval(inf.read())

# Parameter initialization
features_per_node = 9
state_size = features_per_node*21 * 2
action_size = 5
action_dict = dict()
nr_trials_per_test = 100
test_results = []
test_times = []
test_dones = []
# Load agent
#agent = Agent(state_size, action_size, "FC", 0)
#agent.qnetwork_local.load_state_dict(torch.load('./torch_training/Nets/avoid_checkpoint30000.pth'))
agent = RandomAgent(state_size, action_size)
start_time_scoring = time.time()
for test_nr in parameters:
    current_parameters = parameters[test_nr]
    print('\nRunning {} with (x_dim,ydim) = ({},{}) and {} Agents.'.format(test_nr,current_parameters[0],current_parameters[1],current_parameters[2]))
    # Reset all measurements
    time_obs = deque(maxlen=2)
    test_scores = []

    tot_dones = 0
    tot_test_score = 0

    # Reset environment
    random.seed(current_parameters[3])
    np.random.seed(current_parameters[3])
    nr_paths = max(2,current_parameters[2] + int(0.5*current_parameters[2]))
    min_dist = int(min([current_parameters[0], current_parameters[1]])*0.75)
    env = RailEnv(width=current_parameters[0],
                  height=current_parameters[1],
                  rail_generator=complex_rail_generator(nr_start_goal=nr_paths, nr_extra=5, min_dist=min_dist, max_dist=99999,
                                                        seed=current_parameters[3]),
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  number_of_agents=current_parameters[2])
    max_steps = max_steps = int(3 * (env.height + env.width))
    agent_obs = [None] * env.get_num_agents()
    env_renderer = RenderTool(env, gl="PILSVG", )
    printProgressBar(0, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    start = time.time()
    for trial in range(nr_trials_per_test):
        # Reset the env
        printProgressBar(trial+1, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
        obs = env.reset(True, True)
        #env_renderer.set_new_rail()
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=9,
                                                    current_depth=0)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        for i in range(2):
            time_obs.append(obs)

        for a in range(env.get_num_agents()):
            agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

        # Run episode
        trial_score = 0
        for step in range(max_steps):

            for a in range(env.get_num_agents()):

                action = agent.act(agent_obs[a], eps=0)
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree(tree=np.array(next_obs[a]), num_features_per_node=features_per_node,
                                                        current_depth=0)
                data = norm_obs_clip(data)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
            time_obs.append(next_obs)
            for a in range(env.get_num_agents()):
                agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))
                trial_score += all_rewards[a] / env.get_num_agents()
            if done['__all__']:
                tot_dones += 1
                break
        test_scores.append(trial_score / max_steps)
    end = time.time()
    comp_time = end-start
    tot_test_score = np.mean(test_scores)
    test_results.append(tot_test_score)
    test_times.append(comp_time)
    test_dones.append(tot_dones/nr_trials_per_test*100)
end_time_scoring = time.time()
tot_test_time = end_time_scoring-start_time_scoring
test_idx = 0
print('-----------------------------------------------')
print('                     RESULTS')
print('-----------------------------------------------')
for test_nr in parameters:
    print('{} score was = {:.3f} with {:.2f}% environments solved. Test took {} Seconds to complete.'.format(test_nr,
                                                                                                             test_results[test_idx],test_dones[test_idx],test_times[test_idx]))
    test_idx += 1
print('Total scoring duration was', tot_test_time)