## Examples of scripts to train agents in the Flatland environment.


# Torch Training
The `torch_training` folder shows an example of how to train agents with a DQN implemented in pytorch.
In the links below you find introductions to training an agent on Flatland:

- Training an agent for navigation ([Introduction](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/Getting_Started_Training.md))
- Training multiple agents to avoid conflicts ([Introduction](https://gitlab.aicrowd.com/flatland/baselines/blob/master/torch_training/Multi_Agent_Training_Intro.md)) 

Use this introductions to get used to the Flatland environment. Then build your own predictors, observations and agents to improve the performance even more and solve the most complex environments of the challenge.

With the above introductions you will solve tasks like these and even more...

![Conflict_Avoidance](https://i.imgur.com/AvBHKaD.gif)


# RLLib Training
The `RLLib_training` folder shows an example of how to train agents with  algorithm from implemented in the RLLib library available at: <https://github.com/ray-project/ray/tree/master/python/ray/rllib>

# Sequential Agent
This is a very simple baseline to show you have the `complex_level_generator` generates feasible network configurations.
If you run the `run_test.py` file you will see a simple agent that solves the level by sequentially running each agent along its shortest path.

Here you see it in action:

![Sequential_Agent](https://i.imgur.com/DsbG6zK.gif)