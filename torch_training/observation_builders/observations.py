"""
Collection of environment-specific ObservationBuilder.
"""
import pprint
from collections import deque

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position


class TreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """

    def __init__(self, max_depth, predictor=None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 9
        # Compute the size of the returned observation vector
        size = 0
        pow4 = 1
        for i in range(self.max_depth + 1):
            size += pow4
            pow4 *= 4
        self.observation_dim = 9
        self.observation_space = [size * self.observation_dim]
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.agents_previous_reset = None
        self.tree_explored_actions = [1, 2, 3, 0]
        self.tree_explorted_actions_char = ['L', 'F', 'R', 'B']
        self.distance_map = None
        self.distance_map_computed = False

    def reset(self):
        agents = self.env.agents
        nb_agents = len(agents)
        compute_distance_map = True
        if self.agents_previous_reset is not None and nb_agents == len(self.agents_previous_reset):
            compute_distance_map = False
            for i in range(nb_agents):
                if agents[i].target != self.agents_previous_reset[i].target:
                    compute_distance_map = True
        # Don't compute the distance map if it was loaded
        if self.agents_previous_reset is None and self.distance_map is not None:
            self.location_has_target = {tuple(agent.target): 1 for agent in agents}
            compute_distance_map = False

        if compute_distance_map:
            self._compute_distance_map()

        self.agents_previous_reset = agents

    def _compute_distance_map(self):
        agents = self.env.agents
        # For testing only --> To assert if a distance map need to be recomputed.
        self.distance_map_computed = True
        nb_agents = len(agents)
        self.distance_map = np.inf * np.ones(shape=(nb_agents,
                                                    self.env.height,
                                                    self.env.width,
                                                    4))
        self.max_dist = np.zeros(nb_agents)
        self.max_dist = [self._distance_map_walker(agent.target, i) for i, agent in enumerate(agents)]
        # Update local lookup table for all agents' target locations
        self.location_has_target = {tuple(agent.target): 1 for agent in agents}

    def _distance_map_walker(self, position, target_nr):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction`
        nodes_queue = deque(self._get_and_update_neighbors(position, target_nr, 0, enforce_target_direction=-1))

        # BFS from target `position` to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {(position[0], position[1], 0), (position[0], position[1], 1), (position[0], position[1], 2),
                   (position[0], position[1], 3)}

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors((node[0], node[1]), target_nr, node[3], node[2])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, position, target_nr, current_distance, enforce_target_direction=-1):
        """
        Utility function used by _distance_map_walker to perform a BFS walk over the rail, filling in the
        minimum distances from each target cell.
        """
        neighbors = []

        possible_directions = [0, 1, 2, 3]
        if enforce_target_direction >= 0:
            # The agent must land into the current cell with orientation `enforce_target_direction`.
            # This is only possible if the agent has arrived from the cell in the opposite direction!
            possible_directions = [(enforce_target_direction + 2) % 4]

        for neigh_direction in possible_directions:
            new_cell = get_new_position(position, neigh_direction)

            if new_cell[0] >= 0 and new_cell[0] < self.env.height and new_cell[1] >= 0 and new_cell[1] < self.env.width:

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell` to the current cell possible?
                    is_valid = self.env.rail.get_transition((new_cell[0], new_cell[1], agent_orientation),
                                                            desired_movement_from_new_cell)

                    if is_valid:
                        """
                        # TODO: check that it works with deadends! -- still bugged!
                        movement = desired_movement_from_new_cell
                        if isNextCellDeadEnd:
                            movement = (desired_movement_from_new_cell+2) % 4
                        """
                        new_distance = min(self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation],
                                           current_distance + 1)
                        neighbors.append((new_cell[0], new_cell[1], agent_orientation, new_distance))
                        self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation] = new_distance

        return neighbors

    def get_many(self, handles=None):
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get(custom_args={'distance_map': self.distance_map})
            if self.predictions:

                for t in range(len(self.predictions[0])):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        observations = {}
        for h in handles:
            observations[h] = self.get(h)
        return observations

    def get(self, handle):
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is:
            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as:
            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1: if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2: if another agents target is detected the distance in number of cells from the agents current locaiton
            is stored

        #3: if another agent is detected the distance in number of cells from current agent position is stored.

        #4: possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5: if an not usable switch (for agent) is detected we store the distance.

        #6: This feature stores the distance in number of cells to the next branching  (current node)

        #7: minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8: agent in the same direction
            n = number of agents present same direction
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9: agent in the opposite direction
            n = number of agents present other direction than myself (so conflict)
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself




        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target].
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        # Update local lookup table for all agents' positions
        self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents}
        self.location_has_agent_direction = {tuple(agent.position): agent.direction for agent in self.env.agents}
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index
        possible_transitions = self.env.rail.get_transitions(*agent.position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Root node - current position
        observation = [0, 0, 0, 0, 0, 0, self.distance_map[(handle, *agent.position, agent.direction)], 0, 0]

        visited = set()
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent.position, branch_direction)
                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                observation = observation + branch_observation
                visited = visited.union(branch_visited)
            else:
                # add cells filled with infinity if no transition is possible
                observation = observation + [-np.inf] * self._num_cells_to_fill_in(self.max_depth)
        self.env.dev_obs_dict[handle] = visited
        return observation

    def _num_cells_to_fill_in(self, remaining_depth):
        """Computes the length of observation vector: sum_{i=0,depth-1} 2^i * observation_dim."""
        num_observations = 0
        pow4 = 1
        for i in range(remaining_depth):
            num_observations += pow4
            pow4 *= 4
        return num_observations * self.observation_dim

    def _explore_branch(self, handle, position, direction, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """
        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = set()
        agent = self.env.agents[handle]
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0

        num_steps = 1
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                if self.location_has_agent_direction[position] != direction:
                    # Cummulate the number of agents on branch with other direction
                    other_agent_opposite_direction += 1

            # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            if self.predictor and num_steps < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:
                    pre_step = max(0, tot_dist - 1)
                    post_step = min(self.max_prediction_depth - 1, tot_dist + 1)

                    # Look for conflicting paths at distance num_step
                    if int_position in np.delete(self.predicted_pos[tot_dist], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[tot_dist] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[tot_dist][ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.dones[ca] and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           tot_dist,
                           0,
                           other_agent_same_direction,
                           other_agent_opposite_direction
                           ]

        elif last_is_terminal:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           np.inf,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction
                           ]

        else:
            observation = [own_target_encountered,
                           other_target_encountered,
                           other_agent_encountered,
                           potential_conflict,
                           unusable_switch,
                           tot_dist,
                           self.distance_map[handle, position[0], position[1], direction],
                           other_agent_same_direction,
                           other_agent_opposite_direction,
                           ]
        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for branch_direction in [(direction + 4 + i) % 4 for i in range(-1, 3)]:
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                observation = observation + branch_observation
                if len(branch_visited) != 0:
                    visited = visited.union(branch_visited)
            else:
                # no exploring possible, add just cells with infinity
                observation = observation + [-np.inf] * self._num_cells_to_fill_in(self.max_depth - depth)

        return observation, visited

    def util_print_obs_subtree(self, tree):
        """
        Utility function to pretty-print tree observations returned by this object.
        """
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.unfold_observation_tree(tree))

    def unfold_observation_tree(self, tree, current_depth=0, actions_for_display=True):
        """
        Utility function to pretty-print tree observations returned by this object.
        """
        if len(tree) < self.observation_dim:
            return

        depth = 0
        tmp = len(tree) / self.observation_dim - 1
        pow4 = 4
        while tmp > 0:
            tmp -= pow4
            depth += 1
            pow4 *= 4

        unfolded = {}
        unfolded[''] = tree[0:self.observation_dim]
        child_size = (len(tree) - self.observation_dim) // 4
        for child in range(4):
            child_tree = tree[(self.observation_dim + child * child_size):
                              (self.observation_dim + (child + 1) * child_size)]
            observation_tree = self.unfold_observation_tree(child_tree, current_depth=current_depth + 1)
            if observation_tree is not None:
                if actions_for_display:
                    label = self.tree_explorted_actions_char[child]
                else:
                    label = self.tree_explored_actions[child]
                unfolded[label] = observation_tree
        return unfolded

    def _set_env(self, env):
        self.env = env
        if self.predictor:
            self.predictor._set_env(self.env)
