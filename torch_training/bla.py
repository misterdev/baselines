import getopt
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from importlib_resources import path

import torch_training.Nets
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import norm_obs_clip, split_tree

print("multi_agent_trainging.py (1)")
print("bla")