from baselines.RailEnvRLLibWrapper import RailEnvRLLibWrapper
import gym


from flatland.envs.generators import complex_rail_generator

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

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
    env = RailEnv(width=10,
                  height=10,
                  rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
                  number_of_agents=1)
    """
    env_config = {"width":config['map_width'],
                  "height":config['map_height'],
                  "rail_generator":complex_rail_generator(nr_start_goal=50, min_dist=5, max_dist=99999, seed=0),
                  "number_of_agents":config['n_agents']}
    """
    env = RailEnv(width=20,
                  height=20,
                  rail_generator=rail_from_list_of_saved_GridTransitionMap_generator(
                          ['../notebooks/temp.npy']),
                  number_of_agents=3)

    """



    # Example generate a random rail
    # env = RailEnvRLLibWrapper(width=config['map_width'], height=config['map_height'],
    #               rail_generator=complex_rail_generator(nr_start_goal=config["n_agents"], nr_extra=20, min_dist=12),
    #               number_of_agents=config["n_agents"])

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

    # agent_config["num_workers"] = 0
    # agent_config["num_cpus_per_worker"] = 10
    # agent_config["num_gpus"] = 0.5
    # agent_config["num_gpus_per_worker"] = 0.5
    # agent_config["num_cpus_for_driver"] = 1
    # agent_config["num_envs_per_worker"] = 10
    agent_config["env_config"] = env_config
    agent_config["batch_mode"] = "complete_episodes"

    ppo_trainer = PPOTrainer(env=RailEnvRLLibWrapper, config=agent_config)

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
    dir = 'baselines/grid_search_configs/n_agents_grid_search'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)
