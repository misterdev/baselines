import random
from collections import deque

import numpy as np
import torch
from importlib_resources import path

import torch_training.Nets
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import norm_obs_clip, split_tree

random.seed(1)
np.random.seed(1)
"""
file_name = "./railway/complex_scene.pkl"
env = RailEnv(width=10,
              height=20,
              rail_generator=rail_from_file(file_name),
              obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()))
x_dim = env.width
y_dim = env.height
"""

# Parameters for the Environment
x_dim = 20
y_dim = 20
n_agents = 1
n_goals = 5
min_dist = 5

# We are training an Agent using the Tree Observation with depth 2
observation_builder = TreeObsForRailEnv(max_depth=2)

# Use a the malfunction generator to break agents from time to time
stochastic_data = {'prop_malfunction': 0.1,  # Percentage of defective agents
                   'malfunction_rate': 30,  # Rate of malfunction occurence
                   'min_duration': 3,  # Minimal duration of malfunction
                   'max_duration': 20  # Max duration of malfunction
                   }

# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2)

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=sparse_rail_generator(num_cities=5,
                                                   # Number of cities in map (where train stations are)
                                                   num_intersections=4,
                                                   # Number of intersections (no start / target)
                                                   num_trainstations=10,  # Number of possible start/targets on map
                                                   min_node_dist=3,  # Minimal distance of nodes
                                                   node_radius=2,  # Proximity of stations to city center
                                                   num_neighb=3,
                                                   # Number of connections to other cities/intersections
                                                   seed=15,  # Random seed
                                                   grid_mode=True,
                                                   enhance_intersection=False
                                                   ),
              schedule_generator=sparse_schedule_generator(speed_ration_map),
              number_of_agents=n_agents,
              stochastic_data=stochastic_data,  # Malfunction data generator
              obs_builder_object=TreeObservation)
env.reset(True, True)

observation_helper = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PILSVG", )
num_features_per_node = env.obs_builder.observation_dim
handle = env.get_agent_handles()
features_per_node = 9
state_size = features_per_node * 85 * 2
action_size = 5

# We set the number of episodes we would like to train on
if 'n_trials' not in locals():
    n_trials = 60000
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
with path(torch_training.Nets, "avoid_checkpoint49700.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

record_images = False
frame_step = 0

for trials in range(1, n_trials + 1):

    # Reset environment
    obs = env.reset(True, True)

    env_renderer.set_new_rail()
    obs_original = obs.copy()
    final_obs = obs.copy()
    final_obs_next = obs.copy()
    for a in range(env.get_num_agents()):
        data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=num_features_per_node,
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

    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

        if record_images:
            env_renderer.gl.saveImage("./Images/flatland_frame_{:04d}.bmp".format(frame_step))
            frame_step += 1

        # Action
        for a in range(env.get_num_agents()):
            # action = agent.act(np.array(obs[a]), eps=eps)
            action = agent.act(agent_obs[a], eps=0)
            action_dict.update({a: action})
        # Environment step

        next_obs, all_rewards, done, _ = env.step(action_dict)
        # print(all_rewards,action)
        obs_original = next_obs.copy()
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                    num_features_per_node=num_features_per_node,
                                                    current_depth=0)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
        time_obs.append(next_obs)
        for a in range(env.get_num_agents()):
            agent_next_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))
        agent_obs = agent_next_obs.copy()
        if done['__all__']:
            break
