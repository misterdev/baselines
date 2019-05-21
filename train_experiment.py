from baselines.RailEnvRLLibWrapper import RailEnvRLLibWrapper
import gym


from flatland.envs.generators import complex_rail_generator


# Import PPO trainer: we can replace these imports by any other trainer from RLLib.
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer as Trainer
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph as PolicyGraph
from baselines.CustomPPOPolicyGraph import CustomPPOPolicyGraph as PolicyGraph

from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from baselines.CustomPreprocessor import CustomPreprocessor


import ray
import numpy as np

from ray.tune.logger import UnifiedLogger
import tempfile

import gin

from ray import tune

from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
gin.external_configurable(TreeObsForRailEnv)
gin.external_configurable(GlobalObsForRailEnv)

from ray.rllib.models.preprocessors import TupleFlatteningPreprocessor

ModelCatalog.register_custom_preprocessor("tree_obs_prep", CustomPreprocessor)
ModelCatalog.register_custom_preprocessor("global_obs_prep", TupleFlatteningPreprocessor)
ray.init()#object_store_memory=150000000000, redis_max_memory=30000000000)


def train(config, reporter):
    print('Init Env')

    set_seed(config['seed'], config['seed'], config['seed'])

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

    # Example configuration to generate a random rail
    env_config = {"width": config['map_width'],
                  "height": config['map_height'],
                  "rail_generator": complex_rail_generator,
                  "number_of_agents": config['n_agents'],
                  "seed": config['seed'],
                  "obs_builder": config['obs_builder']}

    # Observation space and action space definitions
    if isinstance(config["obs_builder"], TreeObsForRailEnv):
        obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(105,))
        preprocessor = "tree_obs_prep"

    elif isinstance(config["obs_builder"], GlobalObsForRailEnv):
        obs_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 16)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 3)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 4)),
            gym.spaces.Box(low=0, high=1, shape=(4,))))
        preprocessor = "global_obs_prep"

    else:
        raise ValueError("Undefined observation space")


    act_space = gym.spaces.Discrete(4)

    # Dict with the different policies to train
    policy_graphs = {
        config['policy_folder_name'].format(**locals()): (PolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return config['policy_folder_name'].format(**locals())


    # Trainer configuration
    trainer_config = DEFAULT_CONFIG.copy()
    trainer_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": preprocessor}
    trainer_config['multiagent'] = {"policy_graphs": policy_graphs,
                                  "policy_mapping_fn": policy_mapping_fn,
                                  "policies_to_train": list(policy_graphs.keys())}
    trainer_config["horizon"] = config['horizon']

    trainer_config["num_workers"] = 0
    trainer_config["num_cpus_per_worker"] = 3
    trainer_config["num_gpus"] = 0
    trainer_config["num_gpus_per_worker"] = 0
    trainer_config["num_cpus_for_driver"] = 1
    trainer_config["num_envs_per_worker"] = 1
    trainer_config["env_config"] = env_config
    trainer_config["batch_mode"] = "complete_episodes"
    trainer_config['simple_optimizer'] = True
    trainer_config['postprocess_inputs'] = True

    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = config['policy_folder_name'].format(**locals())
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    trainer = Trainer(env=RailEnvRLLibWrapper, config=trainer_config, logger_creator=logger)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print(pretty_print(trainer.train()))

        if i % config['save_every'] == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

        reporter(num_iterations_trained=trainer._iteration)


@gin.configurable
def run_experiment(name, num_iterations, n_agents, hidden_sizes, save_every,
                    map_width, map_height, horizon, policy_folder_name, local_dir, obs_builder, seed):

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
                "horizon": horizon,  # Max number of time steps
                'policy_folder_name': policy_folder_name,
                "obs_builder": obs_builder,
                "seed": seed
                },
        resources_per_trial={
            "cpu": 2,
            "gpu": 0.0
        },
        local_dir=local_dir
    )


if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/home/guillaume/EPFL/Master_Thesis/flatland/baselines/experiment_configs/observation_benchmark'  # To Modify
    gin.parse_config_file(dir + '/config.gin')
    run_experiment(local_dir=dir)
