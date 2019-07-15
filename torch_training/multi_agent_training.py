# Import packages for plotting and system
import getopt
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
# Import Flatland/ Observations and Predictors
from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from importlib_resources import path

# Import Torch and utility functions to normalize observation
import torch_training.Nets
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import norm_obs_clip, split_tree


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_episodes="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_episodes>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_episodes'):
            n_episodes = int(arg)

    ## Initialize the random
    random.seed(1)
    np.random.seed(1)

    # Initialize a random map with a random number of agents
    x_dim = np.random.randint(8, 20)
    y_dim = np.random.randint(8, 20)
    n_agents = np.random.randint(3, 8)
    n_goals = n_agents + np.random.randint(0, 3)
    min_dist = int(0.75 * min(x_dim, y_dim))
    tree_depth = 3
    print("main2")

    """
     Get an observation builder and predictor:
     The predictor will always predict the shortest path from the current location of the agent.
     This is used to warn for potential conflicts --> Should be enhanced to get better performance!
    """
    predictor = ShortestPathPredictorForRailEnv()
    observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=predictor)

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=0),
                  obs_builder_object=observation_helper,
                  number_of_agents=n_agents)
    env.reset(True, True)

    handle = env.get_agent_handles()
    num_features_per_node = env.obs_builder.observation_dim
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_episodes' not in locals():
        n_episodes = 60000

    # Set max number of steps per episode as well as other training relevant parameter
    max_steps = int(3 * (env.height + env.width))
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.9995
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    observation_radius = 10

    # Initialize the agent
    agent = Agent(state_size, action_size, "FC", 0)

    # Here you can pre-load an agent
    if False:
        with path(torch_training.Nets, "avoid_checkpoint30000.pth") as file_in:
            agent.qnetwork_local.load_state_dict(torch.load(file_in))

    # Do training over n_episodes
    for episodes in range(1, n_episodes + 1):
        """
        Training Curriculum: In order to get good generalization we change the number of agents
        and the size of the levels every 50 episodes.
        """
        if episodes % 50 == 0:
            x_dim = np.random.randint(8, 20)
            y_dim = np.random.randint(8, 20)
            n_agents = np.random.randint(3, 8)
            n_goals = n_agents + np.random.randint(0, 3)
            min_dist = int(0.75 * min(x_dim, y_dim))
            env = RailEnv(width=x_dim,
                          height=y_dim,
                          rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                                max_dist=99999,
                                                                seed=0),
                          obs_builder_object=observation_helper,
                          number_of_agents=n_agents)

            # Adjust the parameters according to the new env.
            max_steps = int(3 * (env.height + env.width))
            agent_obs = [None] * env.get_num_agents()
            agent_next_obs = [None] * env.get_num_agents()

        # Reset environment
        obs = env.reset(True, True)

        # Setup placeholder for finals observation of a single agent. This is necessary because agents terminate at
        # different times during an episode
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree(tree=np.array(obs[a]), num_features_per_node=num_features_per_node,
                                                    current_depth=0)
            data = norm_obs_clip(data, fixed_radius=observation_radius)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            # Build agent specific observations and normalize
            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                        num_features_per_node=num_features_per_node, current_depth=0)
                data = norm_obs_clip(data, fixed_radius=observation_radius)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                agent_next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if done[a]:
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: action_dict[a]})
                if not done[a]:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])
                score += all_rewards[a] / env.get_num_agents()

            # Copy observation
            agent_obs = agent_next_obs.copy()

            if done['__all__']:
                env_done = 1
                for a in range(env.get_num_agents()):
                    agent.step(final_obs[a], final_action_dict[a], all_rewards[a], final_obs_next[a], done[a])
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        done_window.append(env_done)
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                episodes,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        if episodes % 100 == 0:
            print(
                '\rTraining {} Agents.\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(),
                    episodes,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps,
                    action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './Nets/avoid_checkpoint' + str(episodes) + '.pth')
            action_prob = [1] * action_size
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
