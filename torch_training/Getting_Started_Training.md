# How to train an Agent on Flatland
Quick introduction on how to train a simple DQN agent using Flatland and Pytorch. At the end of this Tutorial you should be able to train a single agent to navigate in Flatland.
We use the `training_navigation.py` file to train a simple agent with the tree observation to solve the navigation task.

## Actions in Flatland
Flatland is a railway simulation. Thus the actions of an agent are strongly limited to the railway network. This means that in many cases not all actions are valid.
The possible actions of an agent are

- 0 *Do Nothing*:  If the agent is moving it continues moving, if it is stopped id stays stopped
- 1 *Deviate Left*: This action is only valid at cells where the agent can change direction towards left. If action is chosen, the left transition and a rotation of the agent orientation to the left is executed. If the agent is stopped at any position, this action will cause it to start moving in any cell where forward or left is allowed!
- 2 *Go Forward*: This action will start the agent when stopped. At switches this will chose the forward direction.
- 3 *Deviate Right*: Exactly the same as deviate left but for right turns.
- 4 *Stop*: This action causes the agent to stop, this is necessary to avoid conflicts in multi agent setups (Not needed for navigation).

## Tree Observation
Flatland offers 3 basic observations from the beginning. We encourage you to develop your own observations that are better suited for this specific task.
For the navigation training we start with the Tree Observation as agents will learn the task very quickly using this observation.
The tree observation exploits the fact that a railway network is a graph and thus the obersvation is only built along allowed transitions in the graph.

Here is a small example of a railway network with an agent in the top left corner. The tree observation is build by following the allowed transitions for that agent.

.. image:: https://i.imgur.com/utqMx08.png
