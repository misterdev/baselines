from collections import deque
from sys import path

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from dueling_double_dqn import Agent

import torch_training.Nets
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import norm_obs_clip, split_tree

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
env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                    max_dist=99999,
                                                    seed=0),
              obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()),
              number_of_agents=n_agents)
env.reset(True, True)
file_load = False
"""

"""
observation_helper = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PILSVG", )
handle = env.get_agent_handles()
features_per_node = 9
state_size = features_per_node * 85 * 2
action_size = 5
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
with path(torch_training.Nets, "avoid_checkpoint30000.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

demo = True
record_images = False

for trials in range(1, n_trials + 1):

    if trials % 50 == 0 and not demo:
        x_dim = np.random.randint(8, 20)
        y_dim = np.random.randint(8, 20)
        n_agents = np.random.randint(3, 8)
        n_goals = n_agents + np.random.randint(0, 3)
        min_dist = int(0.75 * min(x_dim, y_dim))
        env = RailEnv(width=x_dim,
                      height=y_dim,
                      rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                            max_dist=99999,
                                                            seed=0),
                      obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()),
                      number_of_agents=n_agents)
        env.reset(True, True)
        max_steps = int(3 * (env.height + env.width))
        agent_obs = [None] * env.get_num_agents()
        agent_next_obs = [None] * env.get_num_agents()
    # Reset environment
    if file_load:
        obs = env.reset(False, False)
    else:
        obs = env.reset(True, True)
    if demo:
        env_renderer.set_new_rail()
    obs_original = obs.copy()
    final_obs = obs.copy()
    final_obs_next = obs.copy()
    for a in range(env.get_num_agents()):
        data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=features_per_node,
                                                current_depth=0)
        data = norm_obs_clip(data)
        distance = norm_obs_clip(distance)
        agent_data = np.clip(agent_data, -1, 1)
        obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
        agent_data = env.agents[a]
        speed = 1  # np.random.randint(1,5)
        agent_data.speed_data['speed'] = 1. / speed

    for i in range(2):
        time_obs.append(obs)
    # env.obs_builder.util_print_obs_subtree(tree=obs[0], num_elements_per_node=5)
    for a in range(env.get_num_agents()):
        agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

    score = 0
    env_done = 0
    # Run episode
    for step in range(max_steps):
        if demo:
            env_renderer.renderEnv(show=True, show_observations=True)
            # observation_helper.util_print_obs_subtree(obs_original[0])
            if record_images:
                env_renderer.gl.saveImage("./Images/flatland_frame_{:04d}.bmp".format(step))
        # print(step)
        # Action
        for a in range(env.get_num_agents()):
            if demo:
                eps = 0
            # action = agent.act(np.array(obs[a]), eps=eps)
            action = agent.act(agent_obs[a], eps=eps)
            action_prob[action] += 1
            action_dict.update({a: action})
        # Environment step

        next_obs, all_rewards, done, _ = env.step(action_dict)
        # print(all_rewards,action)
        obs_original = next_obs.copy()
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(next_obs[a]), num_features_per_node=features_per_node,
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
            score += all_rewards[a] / env.get_num_agents()

        agent_obs = agent_next_obs.copy()
        if done['__all__']:
            env_done = 1
            for a in range(env.get_num_agents()):
                agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
            break
    # Epsilon decay
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon

    done_window.append(env_done)
    scores_window.append(score / max_steps)  # save most recent score
    scores.append(np.mean(scores_window))
    dones_list.append((np.mean(done_window)))

    print(
        '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
            env.get_num_agents(), x_dim, y_dim,
            trials,
            np.mean(scores_window),
            100 * np.mean(done_window),
            eps, action_prob / np.sum(action_prob)), end=" ")

    if trials % 100 == 0:
        print(
            '\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(),
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps,
                action_prob / np.sum(action_prob)))
        torch.save(agent.qnetwork_local.state_dict(),
                   './Nets/avoid_checkpoint' + str(trials) + '.pth')
        action_prob = [1] * action_size
plt.plot(scores)
plt.show()
