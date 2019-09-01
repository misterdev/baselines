import getopt
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from dueling_double_dqn import Agent

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    random.seed(1)
    np.random.seed(1)

    # Parameters for the Environment
    x_dim = 20
    y_dim = 20
    n_agents = 1
    n_goals = 5
    min_dist = 5

    # Use a the malfunction generator to break agents from time to time
    stochastic_data = {'prop_malfunction': 0.0,  # Percentage of defective agents
                       'malfunction_rate': 30,  # Rate of malfunction occurence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 20  # Max duration of malfunction
                       }

    # Custom observation builder
    TreeObservation = TreeObsForRailEnv(max_depth=2)

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 1.,  # Fast passenger train
                        1. / 2.: 0.0,  # Fast freight train
                        1. / 3.: 0.0,  # Slow commuter train
                        1. / 4.: 0.0}  # Slow freight train

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(num_cities=5,
                                                       # Number of cities in map (where train stations are)
                                                       num_intersections=4,
                                                       # Number of intersections (no start / target)
                                                       num_trainstations=10,  # Number of possible start/targets on map
                                                       min_node_dist=3,  # Minimal distance of nodes
                                                       node_radius=2,  # Proximity of stations to city center
                                                       num_neighb=3,
                                                       # Number of connections to other cities/intersections
                                                       seed=15,  # Random seed
                                                       grid_mode=True,
                                                       enhance_intersection=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=TreeObservation)
    env.reset(True, True)

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG", )

    # Given the depth of the tree observation and the number of features per node we get the following state_size
    num_features_per_node = env.obs_builder.observation_dim
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes

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
        register_action_state = np.zeros(env.get_num_agents(), dtype=bool)
        final_obs = agent_obs.copy()
        final_obs_next = agent_next_obs.copy()

        # Build agent specific observations
        for a in range(env.get_num_agents()):
            agent_obs[a] = agent_obs[a] = normalize_observation(obs[a], observation_radius=10)

        # Reset score and done
        score = 0
        env_done = 0


        # Run episode
        for step in range(max_steps):

            # Action
            for a in range(env.get_num_agents()):
                if env.agents[a].speed_data['position_fraction'] == 0.:
                    register_action_state[a] = True
                else:
                    register_action_state[a] = False

                action = agent.act(agent_obs[a], eps=eps)
                action_prob[action] += 1
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = env.step(action_dict)

            # Build agent specific observations and normalize
            for a in range(env.get_num_agents()):
                agent_next_obs[a] = normalize_observation(next_obs[a], observation_radius=10)

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                if done[a]:
                    final_obs[a] = agent_obs[a].copy()
                    final_obs_next[a] = agent_next_obs[a].copy()
                    final_action_dict.update({a: action_dict[a]})
                if not done[a] and register_action_state[a]:
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
        tasks_finished = 0
        for _idx in range(env.get_num_agents()):
            if done[_idx] == 1:
                tasks_finished += 1
        done_window.append(tasks_finished / env.get_num_agents())
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
                                                          num_features_per_node=num_features_per_node,
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
        env_renderer.render_env(show=True, show_observations=False)

        # Chose the actions
        for a in range(env.get_num_agents()):
            eps = 0
            action = agent.act(agent_obs[a], eps=eps)
            action_dict.update({a: action})

        # Environment step
        next_obs, all_rewards, done, _ = env.step(action_dict)

        for a in range(env.get_num_agents()):
            rail_data, distance_data, agent_data = split_tree(tree=np.array(next_obs[a]),
                                                              num_features_per_node=num_features_per_node,
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
