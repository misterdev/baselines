from baselines.RailEnvRLLibWrapper import RailEnvRLLibWrapper
import random
import gym


from flatland.envs.generators import complex_rail_generator

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.rllib.models.preprocessors import Preprocessor


import ray
import numpy as np

import gin

from ray import tune


class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (105,)

    def transform(self, observation):
        return observation  # return the preprocessed observation


ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ray.init()


def train(config, reporter):
    print('Init Env')

    env_name = f"rail_env_{config['n_agents']}"  # To modify if different environments configs are explored.

    # Example generate a rail given a manual specification,
    # a map of tuples (cell_type, rotation)
    transition_probability = [0.5,  # empty cell - Case 0
                              1.0,  # Case 1 - straight
                              1.0,  # Case 2 - simple switch
                              0.3,  # Case 3 - diamond drossing
                              0.5,  # Case 4 - single slip
                              0.5,  # Case 5 - double slip
                              0.2,  # Case 6 - symmetrical
                              0.0]  # Case 7 - dead end



    # Example generate a random rail
    env = RailEnvRLLibWrapper(width=config['map_width'], height=config['map_height'],
                  rail_generator=complex_rail_generator(nr_start_goal=config["n_agents"], nr_extra=20, min_dist=12),
                  number_of_agents=config["n_agents"])

    register_env(env_name, lambda _: env)

    obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(105,))
    act_space = gym.spaces.Discrete(4)

    # Dict with the different policies to train
    policy_graphs = {
        f"ppo_policy": (PPOPolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return f"ppo_policy"

    agent_config = ppo.DEFAULT_CONFIG.copy()
    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": "my_prep"}
    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                  "policy_mapping_fn": policy_mapping_fn,
                                  "policies_to_train": list(policy_graphs.keys())}
    agent_config["horizon"] = config['horizon']

    ppo_trainer = PPOAgent(env=env_name, config=agent_config)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if i % config['save_every'] == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)

        reporter(num_iterations_trained=ppo_trainer._iteration)


@gin.configurable
def run_grid_search(name, num_iterations, n_agents, hidden_sizes, save_every,
                    map_width, map_height, horizon, local_dir):

    tune.run(
        train,
        name=name,
        stop={"num_iterations_trained": num_iterations},
        config={"n_agents": n_agents,
                "hidden_sizes": hidden_sizes,  # Array containing the sizes of the network layers
                "save_every": save_every,
                "map_width": map_width,
                "map_height": map_height,
                "local_dir": local_dir,
                "horizon": horizon  # Max number of time steps
                },
        resources_per_trial={
            "cpu": 11,
            "gpu": 0.5
        },
        local_dir=local_dir
    )


if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = 'grid_search_configs/n_agents_grid_search'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)
