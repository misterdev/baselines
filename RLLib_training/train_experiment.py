from baselines.RLLib_training.RailEnvRLLibWrapper import RailEnvRLLibWrapper
import gym

import gin

from flatland.envs.generators import complex_rail_generator

# Import PPO trainer: we can replace these imports by any other trainer from RLLib.
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer as Trainer
# from baselines.CustomPPOTrainer import PPOTrainer as Trainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph as PolicyGraph
# from baselines.CustomPPOPolicyGraph import CustomPPOPolicyGraph as PolicyGraph

from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from baselines.RLLib_training.custom_preprocessors import CustomPreprocessor, ConvModelPreprocessor

from baselines.RLLib_training.custom_models import ConvModelGlobalObs


import ray
import numpy as np

from ray.tune.logger import UnifiedLogger
import tempfile

from ray import tune

from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv,\
                                       LocalObsForRailEnv, GlobalObsForRailEnvDirectionDependent

gin.external_configurable(TreeObsForRailEnv)
gin.external_configurable(GlobalObsForRailEnv)
gin.external_configurable(LocalObsForRailEnv)
gin.external_configurable(GlobalObsForRailEnvDirectionDependent)

from ray.rllib.models.preprocessors import TupleFlatteningPreprocessor

ModelCatalog.register_custom_preprocessor("tree_obs_prep", CustomPreprocessor)
ModelCatalog.register_custom_preprocessor("global_obs_prep", TupleFlatteningPreprocessor)
ModelCatalog.register_custom_preprocessor("conv_obs_prep", ConvModelPreprocessor)
ModelCatalog.register_custom_model("conv_model", ConvModelGlobalObs)
ray.init()#object_store_memory=150000000000, redis_max_memory=30000000000)


def train(config, reporter):
    print('Init Env')

    set_seed(config['seed'], config['seed'], config['seed'])

    # Example configuration to generate a random rail
    env_config = {"width": config['map_width'],
                  "height": config['map_height'],
                  "rail_generator": config["rail_generator"],
                  "nr_extra": config["nr_extra"],
                  "number_of_agents": config['n_agents'],
                  "seed": config['seed'],
                  "obs_builder": config['obs_builder']}

    # Observation space and action space definitions
    if isinstance(config["obs_builder"], TreeObsForRailEnv):
        obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=float('inf'), shape=(147,)),
                                     gym.spaces.Box(low=0, high=1, shape=(config['n_agents'],)),
                                     gym.spaces.Box(low=0, high=1, shape=(20, config['n_agents'])),
                                     gym.spaces.Box(low=0, high=float('inf'), shape=(147,)),
                                     gym.spaces.Box(low=0, high=1, shape=(config['n_agents'],)),
                                     gym.spaces.Box(low=0, high=1, shape=(20, config['n_agents']))))
        preprocessor = "tree_obs_prep"

    elif isinstance(config["obs_builder"], GlobalObsForRailEnv):
        obs_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 16)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 8)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 2))))
        if config['conv_model']:
            preprocessor = "conv_obs_prep"
        else:
            preprocessor = "global_obs_prep"

    elif isinstance(config["obs_builder"], GlobalObsForRailEnvDirectionDependent):
        obs_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 16)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 5)),
            gym.spaces.Box(low=0, high=1, shape=(config['map_height'], config['map_width'], 2))))
        if config['conv_model']:
            preprocessor = "conv_obs_prep"
        else:
            preprocessor = "global_obs_prep"

    elif isinstance(config["obs_builder"], LocalObsForRailEnv):
        view_radius = config["obs_builder"].view_radius
        obs_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(2 * view_radius + 1, 2 * view_radius + 1, 16)),
            gym.spaces.Box(low=0, high=1, shape=(2 * view_radius + 1, 2 * view_radius + 1, 2)),
            gym.spaces.Box(low=0, high=1, shape=(2 * view_radius + 1, 2 * view_radius + 1, 4)),
            gym.spaces.Box(low=0, high=1, shape=(4,))))
        preprocessor = "global_obs_prep"

    else:
        raise ValueError("Undefined observation space")


    act_space = gym.spaces.Discrete(5)

    # Dict with the different policies to train
    policy_graphs = {
        config['policy_folder_name'].format(**locals()): (PolicyGraph, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id):
        return config['policy_folder_name'].format(**locals())


    # Trainer configuration
    trainer_config = DEFAULT_CONFIG.copy()
    if config['conv_model']:
        trainer_config['model'] = {"custom_model": "conv_model", "custom_preprocessor": preprocessor}
    else:
        trainer_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": preprocessor}

    trainer_config['multiagent'] = {"policy_graphs": policy_graphs,
                                  "policy_mapping_fn": policy_mapping_fn,
                                  "policies_to_train": list(policy_graphs.keys())}
    trainer_config["horizon"] = config['horizon']

    trainer_config["num_workers"] = 0
    trainer_config["num_cpus_per_worker"] = 11
    trainer_config["num_gpus"] = 0.5
    trainer_config["num_gpus_per_worker"] = 0.5
    trainer_config["num_cpus_for_driver"] = 1
    trainer_config["num_envs_per_worker"] = 6
    trainer_config['entropy_coeff'] = config['entropy_coeff']
    trainer_config["env_config"] = env_config
    trainer_config["batch_mode"] = "complete_episodes"
    trainer_config['simple_optimizer'] = False
    trainer_config['postprocess_inputs'] = True
    trainer_config['log_level'] = 'WARN'
    trainer_config['num_sgd_iter'] = 10
    trainer_config['clip_param'] = 0.2
    trainer_config['kl_coeff'] = config['kl_coeff']
    trainer_config['lambda'] = config['lambda_gae']

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
                   map_width, map_height, horizon, policy_folder_name, local_dir, obs_builder,
                   entropy_coeff, seed, conv_model, rail_generator, nr_extra, kl_coeff, lambda_gae):

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
                "entropy_coeff": entropy_coeff,
                "seed": seed,
                "conv_model": conv_model,
                "rail_generator": rail_generator,
                "nr_extra": nr_extra,
                "kl_coeff": kl_coeff,
                "lambda_gae": lambda_gae
                },
        resources_per_trial={
            "cpu": 12,
            "gpu": 0.5
        },
        local_dir=local_dir
    )


if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/home/guillaume/flatland/baselines/RLLib_training/experiment_configs/env_size_benchmark_3_agents'  # To Modify
    gin.parse_config_file(dir + '/config.gin')
    run_experiment(local_dir=dir)
