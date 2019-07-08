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

![Small_Network](https://i.imgur.com/utqMx08.png)

As we move along the allowed transitions we build up a tree where a new node is created at every cell where the agent has different possibilities (Switch) or the target is reached. It is important to note that the tree observation is always build according to the orientation of the agent at a given node. This means that each node always has 4 branches coming from it in the directions *Left, Forward, Right and Backward*. These are illustrated with different colors in the figure below. The tree is build form the example rail above. Nodes where there are no possibilitis are fill with `-inf` and are not all shown here for simplicity. The tree however, always hase the same number of nodes for a given tree depth.

![Tree_Observation](https://i.imgur.com/VsUQOQz.png)

### Node Information
Each node is filled with information gathered along the path to the node. Currently each node contains 9 features:

- 1: if own target lies on the explored branch the current distance from the agent in number of cells is stored.

- 2: if another agents target is detected the distance in number of cells from the agents current locaiton is stored.

- 3: if another agent is detected the distance in number of cells from current agent position is stored.

- 4: possible conflict detected
    tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the distance in number of cells from current agent position 0 = No other agent reserve the same cell at similar time.

- 5: if an not usable switch (for agent) is detected we store the distance.

- 6: This feature stores the distance in number of cells to the next branching  (current node)

- 7: minimum distance from node to the agent's target given the direction of the agent if this path is chosen

- 8: agent in the same direction
    - n = number of agents present same direction (possible future use: number of other agents in the same direction in this branch)
    - 0 = no agent present same direction

- 9: agent in the opposite direction
    - n = number of agents present other direction than myself
    - 0 = no agent present other direction than myself
