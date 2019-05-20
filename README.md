This repository allows to run Rail Environment multi agent training with the RLLib Library.

It should be clone inside the main flatland repository.

## Installation:
```sh
pip install ray
pip install gin-config
```

To start a grid search on some parameters, you can create a folder containing a config.gin file (see example in `grid_search_configs/n_agents_grid_search/config.gin`.

Then, you can modify the config.gin file path at the end of the `grid_search_train.py` file.

The results will be stored inside the folder, and the learning curves can be visualized in 
tensorboard:

```
tensorboard --logdir=/path/to/foler_containing_config_gin_file
```

## Gin config files

In each config.gin files, all the parameters, except `local_dir` of the `run_experiment` functions have to be specified.
For example, to indicate the number of agents that have to be initialized at the beginning of each simulation, the following line should be added:

```
run_experiment.n_agents = 2
```

If several number of agents have to be explored during the experiment, one can pass the following value to the `n_agents` parameter:

```
run_experiment.n_agents = {"grid_search": [2,5]}
```

which is the way to indicate to the tune library to experiment several values for a parameter.

To reference a class or an object within gin, you should first register it from the `train_experiment.py` script adding the following line:

```
gin.external_configurable(TreeObsForRailEnv)
```

and then a `TreeObsForRailEnv` object can be referenced in the `config.gin` file:

```
run_experiment.obs_builder = {"grid_search": [@TreeObsForRailEnv(), @GlobalObsForRailEnv()]}
TreeObsForRailEnv.max_depth = 2
```

Note that `@TreeObsForRailEnv` references the class, while `@TreeObsForRailEnv()` references instantiates an object of this class.




More documentation on how to use gin-config can be found on the library github repository: https://github.com/google/gin-config
