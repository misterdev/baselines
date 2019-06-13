import os
import random
from collections import deque

import numpy as np
import torch

from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from torch_training.dueling_double_dqn import Agent

random.seed(1)
np.random.seed(1)

__file_dirname__ = os.path.dirname(os.path.realpath(__file__))

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
transition_probability = [15,  # empty cell - Case 0
                          5,  # Case 1 - straight
                          5,  # Case 2 - simple switch
                          1,  # Case 3 - diamond crossing
                          1,  # Case 4 - single slip
                          1,  # Case 5 - double slip
                          1,  # Case 6 - symmetrical
                          0,  # Case 7 - dead end
                          1,  # Case 1b (8)  - simple turn right
                          1,  # Case 1c (9)  - simple turn left
                          1]  # Case 2b (10) - simple switch mirrored

# Example generate a random rail
"""
env = RailEnv(width=20,
              height=20,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=1)

env = RailEnv(width=15,
              height=15,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=10, min_dist=10, max_dist=99999, seed=0),
              number_of_agents=1)



env = RailEnv(width=10,
              height=20)
env.load_resource('torch_training.railway', "complex_scene.pkl")
"""

env = RailEnv(width=20,
              height=20,
              rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999, seed=0),
              obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv()),
              number_of_agents=10)
env.reset(True, True)

env_renderer = RenderTool(env, gl="PILSVG")
handle = env.get_agent_handles()

state_size = 147 * 2
action_size = 5
n_trials = 15000
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
agent.qnetwork_local.load_state_dict(torch.load('./Nets/avoid_checkpoint15000.pth'))

demo = True


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


for trials in range(1, n_trials + 1):

    # Reset environment
    obs = env.reset(True, True)
    if demo:
        env_renderer.set_new_rail()
    final_obs = obs.copy()
    final_obs_next = obs.copy()

    for a in range(env.get_num_agents()):
        data, distance, agent_data = env.obs_builder.split_tree(tree=np.array(obs[a]), num_features_per_node=7,
                                                                current_depth=0)
        data = norm_obs_clip(data)
        distance = norm_obs_clip(distance)
        agent_data = np.clip(agent_data, -1, 1)
        obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
    for i in range(2):
        time_obs.append(obs)
    # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)
    for a in range(env.get_num_agents()):
        agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

    score = 0
    env_done = 0
    # Run episode
    for step in range(env.height * env.width):
        if demo:
            env_renderer.renderEnv(show=True, show_observations=False)
        # print(step)
        # Action
        for a in range(env.get_num_agents()):
            if demo:
                eps = 1
            # action = agent.act(np.array(obs[a]), eps=eps)
            action = agent.act(agent_obs[a], eps=eps)
            action_prob[action] += 1
            action_dict.update({a: action})
        # Environment step

        next_obs, all_rewards, done, _ = env.step(action_dict)
        for a in range(env.get_num_agents()):
            data, distance, agent_data = env.obs_builder.split_tree(tree=np.array(next_obs[a]), num_features_per_node=7,
                                                                    current_depth=0)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        time_obs.append(next_obs)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent_next_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))
            if done[a]:
                final_obs[a] = agent_obs[a].copy()
                final_obs_next[a] = agent_next_obs[a].copy()
                final_action_dict.update({a: action_dict[a]})
            if not demo and not done[a]:
                agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])
            score += all_rewards[a]

        agent_obs = agent_next_obs.copy()
        if done['__all__']:
            env_done = 1
            for a in range(env.get_num_agents()):
                agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
            break
    # Epsilon decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    done_window.append(env_done)
    scores_window.append(score)  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    print(
        '\rTraining {} Agents.\t Episode {}\t Average Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
            env.get_num_agents(),
            trials,
            np.mean(scores_window),
            100 * np.mean(done_window),
            eps, action_prob / np.sum(action_prob)), end=" ")

    if trials % 100 == 0:
        print(
            '\rTraining {} Agents.\t Episode {}\t Average Score: {:.0f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(),
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps,
                action_prob / np.sum(action_prob)))
        torch.save(agent.qnetwork_local.state_dict(),
                   os.path.join(__file_dirname__, 'Nets', 'avoid_checkpoint' + str(trials) + '.pth'))
        action_prob = [1] * action_size
