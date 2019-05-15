This repository allows to run Rail Environment multi agent training with the RLLib Library.
To start a grid search on some parameters, you can create a folder containing a config.gin file.
Then, you can modify the config.gin file path at the end of the grid_search_train.py file.

The results will be stored inside the folder, and the learning curves can be visualized in 
tensorboard: 'tensorboard --logdir=/path/to/foler_containing_config_gin_file'.