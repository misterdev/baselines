import os

import gin
import gym
from flatland.envs.predictions import DummyPredictorForRailEnv, ShortestPathPredictorForRailEnv
from importlib_resources import path
# Import PPO trainer: we can replace these imports by any other trainer from RLLib.
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer as Trainer
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph as PolicyGraph
from ray.rllib.models import ModelCatalog

gin.external_configurable(DummyPredictorForRailEnv)
gin.external_configurable(ShortestPathPredictorForRailEnv)

import ray

from ray.tune.logger import UnifiedLogger
from ray.tune.logger import pretty_print

from RailEnvRLLibWrapper import RailEnvRLLibWrapper
from custom_models import ConvModelGlobalObs
from custom_preprocessors import CustomPreprocessor, ConvModelPreprocessor
import tempfile

from ray import tune

from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv, \
    LocalObsForRailEnv, GlobalObsForRailEnvDirectionDependent

gin.external_configurable(TreeObsForRailEnv)
gin.external_configurable(GlobalObsForRailEnv)
gin.external_configurable(LocalObsForRailEnv)
gin.external_configurable(GlobalObsForRailEnvDirectionDependent)

from ray.rllib.models.preprocessors import TupleFlatteningPreprocessor
import numpy as np

ModelCatalog.register_custom_preprocessor("tree_obs_prep", CustomPreprocessor)
ModelCatalog.register_custom_preprocessor("global_obs_prep", TupleFlatteningPreprocessor)
ModelCatalog.register_custom_preprocessor("conv_obs_prep", ConvModelPreprocessor)
ModelCatalog.register_custom_model("conv_model", ConvModelGlobalObs)
ray.init()  # object_store_memory=150000000000, redis_max_memory=30000000000)

__file_dirname__ = os.path.dirname(os.path.realpath(__file__))


def on_episode_start(info):
    episode = info['episode']
    map_width = info['env'].envs[0].width
    map_height = info['env'].envs[0].height
    episode.horizon = map_width + map_height
    

# def on_episode_step(info):
#     episode = info['episode']
#     print('#########################', episode._agent_reward_history)
#     # print(ds)


def on_episode_end(info):
    episode = info['episode']
    score = 0
    for k, v in episode._agent_reward_history.items():
        score += np.sum(v)
    score /= (len(episode._agent_reward_history) * 3 * episode.horizon)
    episode.custom_metrics["score"] = score


def train(config, reporter):
    print('Init Env')

    set_seed(config['seed'], config['seed'], config['seed'])
    config['map_height'] = config['map_width']

    # Example configuration to generate a random rail
    env_config = {"width": config['map_width'],
                  "height": config['map_height'],
                  "rail_generator": config["rail_generator"],
                  "nr_extra": config["nr_extra"],
                  "number_of_agents": config['n_agents'],
                  "seed": config['seed'],
                  "obs_builder": config['obs_builder'],
                  "min_dist": config['min_dist'],
                  "step_memory": config["step_memory"]}

    # Observation space and action space definitions
    if isinstance(config["obs_builder"], TreeObsForRailEnv):
        obs_space = gym.spaces.Tuple((gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(168,)),) * 2)
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
    trainer_config["horizon"] = 3 * (config['map_width'] + config['map_height'])#config['horizon']

    trainer_config["num_workers"] = 0
    trainer_config["num_cpus_per_worker"] = 7
    trainer_config["num_gpus"] = 0.0
    trainer_config["num_gpus_per_worker"] = 0.0
    trainer_config["num_cpus_for_driver"] = 1
    trainer_config["num_envs_per_worker"] = 1
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
    trainer_config['callbacks'] = {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_end": tune.function(on_episode_end)
        }

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
                   map_width, map_height, policy_folder_name, local_dir, obs_builder,
                   entropy_coeff, seed, conv_model, rail_generator, nr_extra, kl_coeff, lambda_gae,
                   step_memory, min_dist):
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
                'policy_folder_name': policy_folder_name,
                "obs_builder": obs_builder,
                "entropy_coeff": entropy_coeff,
                "seed": seed,
                "conv_model": conv_model,
                "rail_generator": rail_generator,
                "nr_extra": nr_extra,
                "kl_coeff": kl_coeff,
                "lambda_gae": lambda_gae,
                "min_dist": min_dist,
                # "predictor": predictor,
                "step_memory": step_memory
                },
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        },
        verbose=2,
        local_dir=local_dir
    )


if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    # with path('RLLib_training.experiment_configs.n_agents_experiment', 'config.gin') as f:
    #     gin.parse_config_file(f)
    gin.parse_config_file('/mount/SDC/flatland/baselines/RLLib_training/experiment_configs/env_size_benchmark_3_agents/config.gin')
    dir = '/mount/SDC/flatland/baselines/RLLib_training/experiment_configs/env_size_benchmark_3_agents'
    # dir = os.path.join(__file_dirname__, 'experiment_configs', 'experiment_agent_memory')
    run_experiment(local_dir=dir)
