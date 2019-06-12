import random

import gym
import numpy as np
import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from RLLib_training.custom_preprocessors import CustomPreprocessor
from RailEnvRLLibWrapper import RailEnvRLLibWrapper
from flatland.envs.generators import complex_rail_generator

ModelCatalog.register_custom_preprocessor("my_prep", CustomPreprocessor)
ray.init()


def train(config):
    print('Init Env')
    random.seed(1)
    np.random.seed(1)

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
    env_config = {"width": 20,
                  "height": 20,
                  "rail_generator": complex_rail_generator(nr_start_goal=5, min_dist=5, max_dist=99999, seed=0),
                  "number_of_agents": 5}

    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(105,))
    act_space = gym.spaces.Discrete(4)

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy": (PPOPolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return f"ppo_policy"

    agent_config = ppo.DEFAULT_CONFIG.copy()
    agent_config['model'] = {"fcnet_hiddens": [32, 32], "custom_preprocessor": "my_prep"}
    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                  "policy_mapping_fn": policy_mapping_fn,
                                  "policies_to_train": list(policy_graphs.keys())}
    agent_config["horizon"] = 50
    agent_config["num_workers"] = 0
    # agent_config["sample_batch_size"]: 1000
    # agent_config["num_cpus_per_worker"] = 40
    # agent_config["num_gpus"] = 2.0
    # agent_config["num_gpus_per_worker"] = 2.0
    # agent_config["num_cpus_for_driver"] = 5
    # agent_config["num_envs_per_worker"] = 15
    agent_config["env_config"] = env_config
    # agent_config["batch_mode"] = "complete_episodes"

    ppo_trainer = PPOTrainer(env=RailEnvRLLibWrapper, config=agent_config)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # if i % config['save_every'] == 0:
        #     checkpoint = ppo_trainer.save()
        #     print("checkpoint saved at", checkpoint)


train({})
