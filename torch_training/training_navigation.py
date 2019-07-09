import sys
from collections import deque

import getopt
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from dueling_double_dqn import Agent

from flatland.envs.generators import complex_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import norm_obs_clip, split_tree


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n','--n_trials'):
            n_trials = arg

    random.seed(1)
    np.random.seed(1)

    # Parameters for the Environment
    x_dim = 10
    y_dim = 10
    n_agents = 1
    n_goals = 5
    min_dist = 5

    # We are training an Agent using the Tree Observation with depth 2
    observation_builder = TreeObsForRailEnv(max_depth=2)

    # Load the Environment
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=0),
                  obs_builder_object=observation_builder,
                  number_of_agents=n_agents)
    env.reset(True, True)

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG", )

    # Given the depth of the tree observation and the number of features per node we get the following state_size
    features_per_node = 9
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = features_per_node * nr_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_trials' not in locals():
        n_trials = 6000

    # And the max number of steps we want to take per episode
    max_steps = int(3 * (env.height + env.width))

    # Define training parameters
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998

    # And some variables to keep track of the progress
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    time_obs = deque(maxlen=2)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()

    # Now we load a Double dueling DQN agent
    agent = Agent(state_size, action_size, "FC", 0)

    Training = True

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs = env.reset(True, True)
        if not Training:
            env_renderer.set_new_rail()

        # Split the observation tree into its parts and normalize the observation using the utility functions.
        # Build agent specific local observation
        for a in range(env.get_num_agents()):
            rail_data, distance_data, agent_data = split_tree(tree=np.array(obs[a]),
                                                              current_depth=0)
            rail_data = norm_obs_clip(rail_data)
            distance_data = norm_obs_clip(distance_data)
            agent_data = np.clip(agent_data, -1, 1)
            agent_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

        # Reset score and done
        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):

            # Only render when not triaing
            if not Training:
                env_renderer.renderEnv(show=True, show_observations=True)

            # Chose the actions
            for a in range(env.get_num_agents()):
                if not Training:
                    eps = 0

                action = agent.act(agent_obs[a], eps=eps)
                action_dict.update({a: action})

                # Count number of actions takes for statistics
                action_prob[action] += 1

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            for a in range(env.get_num_agents()):
                rail_data, distance_data, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                                  current_depth=0)
                rail_data = norm_obs_clip(rail_data)
                distance_data = norm_obs_clip(distance_data)
                agent_data = np.clip(agent_data, -1, 1)
                agent_next_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):

                # Remember and train agent
                if Training:
                    agent.step(agent_obs[a], action_dict[a], all_rewards[a], agent_next_obs[a], done[a])

                # Update the current score
                score += all_rewards[a] / env.get_num_agents()

            agent_obs = agent_next_obs.copy()
            if done['__all__']:
                env_done = 1
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Store the information about training progress
        done_window.append(env_done)
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        if trials % 100 == 0:
            print(
                '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(), x_dim, y_dim,
                    trials,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps, action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './Nets/navigator_checkpoint' + str(trials) + '.pth')
            action_prob = [1] * action_size

    # Render the trained agent

    # Reset environment
    obs = env.reset(True, True)
    env_renderer.set_new_rail()

    # Split the observation tree into its parts and normalize the observation using the utility functions.
    # Build agent specific local observation
    for a in range(env.get_num_agents()):
        rail_data, distance_data, agent_data = split_tree(tree=np.array(obs[a]),
                                                          current_depth=0)
        rail_data = norm_obs_clip(rail_data)
        distance_data = norm_obs_clip(distance_data)
        agent_data = np.clip(agent_data, -1, 1)
        agent_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

    # Reset score and done
    score = 0
    env_done = 0

    # Run episode
    for step in range(max_steps):
        env_renderer.renderEnv(show=True, show_observations=False)

        # Chose the actions
        for a in range(env.get_num_agents()):
            eps = 0
            action = agent.act(agent_obs[a], eps=eps)
            action_dict.update({a: action})

        # Environment step
        next_obs, all_rewards, done, _ = env.step(action_dict)

        for a in range(env.get_num_agents()):
            rail_data, distance_data, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                              current_depth=0)
            rail_data = norm_obs_clip(rail_data)
            distance_data = norm_obs_clip(distance_data)
            agent_data = np.clip(agent_data, -1, 1)
            agent_next_obs[a] = np.concatenate((np.concatenate((rail_data, distance_data)), agent_data))

        agent_obs = agent_next_obs.copy()
        if done['__all__']:
            break
    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
