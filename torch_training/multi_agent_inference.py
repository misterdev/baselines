import random
from collections import deque

import numpy as np
import torch
from flatland.envs.generators import rail_from_file, complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from importlib_resources import path

import torch_training.Nets
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import normalize_observation

random.seed(3)
np.random.seed(2)

file_name = "./railway/simple_avoid.pkl"
env = RailEnv(width=10,
              height=20,
              rail_generator=rail_from_file(file_name),
              obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()))
x_dim = env.width
y_dim = env.height

"""

x_dim = 18  # np.random.randint(8, 20)
y_dim = 14  # np.random.randint(8, 20)
n_agents = 7  # np.random.randint(3, 8)
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
"""
tree_depth = 3
observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PILSVG", )
handle = env.get_agent_handles()
num_features_per_node = env.obs_builder.observation_dim
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes
action_size = 5

n_trials = 10
observation_radius = 10
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
with path(torch_training.Nets, "avoid_checkpoint46200.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

record_images = False
frame_step = 0

for trials in range(1, n_trials + 1):

    # Reset environment
    obs = env.reset(True, True)

    env_renderer.reset()

    for a in range(env.get_num_agents()):
        agent_obs[a] = normalize_observation(obs[a], observation_radius=10)

    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

        if record_images:
            env_renderer.gl.save_image("./Images/Avoiding/flatland_frame_{:04d}.bmp".format(frame_step))
            frame_step += 1

        # Action
        for a in range(env.get_num_agents()):
            action = agent.act(agent_obs[a], eps=0)
            action_dict.update({a: action})
        # Environment step

        next_obs, all_rewards, done, _ = env.step(action_dict)
        for a in range(env.get_num_agents()):
            agent_obs[a] = normalize_observation(next_obs[a], observation_radius=10)

        if done['__all__']:
            break
