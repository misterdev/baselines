from flatland.envs.rail_env import RailEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from flatland.envs.observations import TreeObsForRailEnv
from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.generators import complex_rail_generator, random_rail_generator
import numpy as np


class RailEnvRLLibWrapper(MultiAgentEnv):

    def __init__(self, config):

        super(MultiAgentEnv, self).__init__()
        if hasattr(config, "vector_index"):
            vector_index = config.vector_index
        else:
            vector_index = 1

        self.predefined_env = False

        if config['rail_generator'] == "complex_rail_generator":
            self.rail_generator = complex_rail_generator(nr_start_goal=config['number_of_agents'], min_dist=5,
                                                          nr_extra=config['nr_extra'], seed=config['seed'] * (1+vector_index))
        elif config['rail_generator'] == "random_rail_generator":
            self.rail_generator = random_rail_generator()
        elif config['rail_generator'] == "load_env":
            self.predefined_env = True

        else:
            raise(ValueError, f'Unknown rail generator: {config["rail_generator"]}')

        set_seed(config['seed'] * (1+vector_index))
        self.env = RailEnv(width=config["width"], height=config["height"],
                number_of_agents=config["number_of_agents"],
                obs_builder_object=config['obs_builder'], rail_generator=self.rail_generator,
                prediction_builder_object=config['predictor'])

        if self.predefined_env:
            self.env.load(config['load_env_path'])
            self.env.load_resource('torch_training.railway', config['load_env_path'])

        self.width = self.env.width
        self.height = self.env.height
        self.step_memory = config["step_memory"]

    def reset(self):
        self.agents_done = []
        if self.predefined_env:
            obs = self.env.reset(False, False)
        else:
            obs = self.env.reset()

        predictions = self.env.predict()
        if predictions != {}:
            # pred_pos is a 3 dimensions array (N_Agents, T_pred, 2) containing x and y coordinates of
            # agents at each time step
            pred_pos = np.concatenate([[x[:, 1:3]] for x in list(predictions.values())], axis=0)
            pred_dir = [x[:, 2] for x in list(predictions.values())]

        o = dict()

        for i_agent in range(len(self.env.agents)):

            if predictions != {}:
                pred_obs = self.get_prediction_as_observation(pred_pos, pred_dir, i_agent)

                agent_id_one_hot = np.zeros(len(self.env.agents))
                agent_id_one_hot[i_agent] = 1
                o[i_agent] = [obs[i_agent], agent_id_one_hot, pred_obs]
            else:
                o[i_agent] = obs[i_agent]

        # needed for the renderer
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict

        if self.step_memory < 2:
            return o
        else:
            self.old_obs = o
            oo = dict()

            for i_agent in range(len(self.env.agents)):
                oo[i_agent] = [o[i_agent], o[i_agent]]
            return oo

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)

        d = dict()
        r = dict()
        o = dict()

        predictions = self.env.predict()
        if predictions != {}:
            # pred_pos is a 3 dimensions array (N_Agents, T_pred, 2) containing x and y coordinates of
            # agents at each time step
            pred_pos = np.concatenate([[x[:, 1:3]] for x in list(predictions.values())], axis=0)
            pred_dir = [x[:, 2] for x in list(predictions.values())]

        for i_agent in range(len(self.env.agents)):
            if i_agent not in self.agents_done:

                if predictions != {}:
                    pred_obs = self.get_prediction_as_observation(pred_pos, pred_dir, i_agent)
                    agent_id_one_hot = np.zeros(len(self.env.agents))
                    agent_id_one_hot[i_agent] = 1
                    o[i_agent] = [obs[i_agent], agent_id_one_hot, pred_obs]
                else:
                    o[i_agent] = obs[i_agent]
                r[i_agent] = rewards[i_agent]
                d[i_agent] = dones[i_agent]

        d['__all__'] = dones['__all__']

        if self.step_memory >= 2:
            oo = dict()

            for i_agent in range(len(self.env.agents)):
                if i_agent not in self.agents_done:
                    oo[i_agent] = [o[i_agent], self.old_obs[i_agent]]

            self.old_obs = o

        for agent, done in dones.items():
            if done and agent != '__all__':
                self.agents_done.append(agent)

        if self.step_memory < 2:
            return o, r, d, infos
        else:
            return oo, r, d, infos

    def get_agent_handles(self):
        return self.env.get_agent_handles()

    def get_num_agents(self):
        return self.env.get_num_agents()

    def get_prediction_as_observation(self, pred_pos, pred_dir, agent_handle):
        '''
        :param pred_pos: pred_pos should be a 3 dimensions array (N_Agents, T_pred, 2) containing x and y
                         predicted coordinates of agents at each time step
        :param pred_dir: pred_dir should be a 2 dimensions array (N_Agents, T_pred) predicted directions
                         of agents at each time step
        :param agent_handle: agent index
        :return: 2 dimensional array (T_pred, N_agents) with value 1 at coord. (t,i) if agent 'agent_handle'
                and agent i are going to meet at time step t.

        Computes prediction of collision that will be added to the observation.
        Allows to the agent to know which other train it is about to meet, and when.
        The id of the other trains are shared, allowing eventually the agents to come
        up with a priority order of trains.
        '''

        pred_obs = np.zeros((len(pred_pos[1]), len(self.env.agents)))

        for time_offset in range(len(pred_pos[1])):

            # We consider a time window of t-1:t+1 to find a collision
            collision_window = list(range(max(time_offset - 1, 0), min(time_offset + 2, len(pred_pos[1]))))

            # coordinate of agent `agent_handle` at time t.
            coord_agent = pred_pos[agent_handle, time_offset, 0] + 1000 * pred_pos[agent_handle, time_offset, 1]

            # x coordinates of all other agents in the time window
            # array of dim (N_Agents, 3), the 3 elements corresponding to x coordinates of the agents
            # at t-1, t, t + 1
            x_coord_other_agents = pred_pos[list(range(agent_handle)) +
                                            list(range(agent_handle + 1,
                                                       len(self.env.agents)))][:, collision_window, 0]

            # y coordinates of all other agents in the time window
            # array of dim (N_Agents, 3), the 3 elements corresponding to y coordinates of the agents
            # at t-1, t, t + 1
            y_coord_other_agents = pred_pos[list(range(agent_handle)) +
                                            list(range(agent_handle + 1, len(self.env.agents)))][
                                   :, collision_window, 1]

            coord_other_agents = x_coord_other_agents + 1000 * y_coord_other_agents

            # collision_info here contains the index of the agent colliding with the current agent and
            # the delta_t at which they visit the same cell (0 for t-1, 1 for t or 2 for t+1)
            for collision_info in np.argwhere(coord_agent == coord_other_agents):
                # If they are on the same cell at the same time, there is a collison in all cases
                if collision_info[1] == 1:
                    pred_obs[time_offset, collision_info[0] + 1 * (collision_info[0] >= agent_handle)] = 1
                elif collision_info[1] == 0:
                    # In this case, the other agent (agent 2) was on the same cell at t-1
                    # There is a collision if agent 2 is at t, on the cell where was agent 1 at t-1
                    coord_agent_1_t_minus_1 = pred_pos[agent_handle, time_offset-1, 0] + \
                                          1000 * pred_pos[agent_handle, time_offset, 1]
                    coord_agent_2_t = coord_other_agents[collision_info[0], 1]
                    if coord_agent_1_t_minus_1 == coord_agent_2_t:
                        pred_obs[time_offset, collision_info[0] + 1 * (collision_info[0] >= agent_handle)] = 1

                elif collision_info[1] == 2:
                    # In this case, the other agent (agent 2) will be on the same cell at t+1
                    # There is a collision if agent 2 is at t, on the cell where will be agent 1 at t+1
                    coord_agent_1_t_plus_1 = pred_pos[agent_handle, time_offset + 1, 0] + \
                                              1000 * pred_pos[agent_handle, time_offset, 1]
                    coord_agent_2_t = coord_other_agents[collision_info[0], 1]
                    if coord_agent_1_t_plus_1 == coord_agent_2_t:
                        pred_obs[time_offset, collision_info[0] + 1 * (collision_info[0] >= agent_handle)] = 1

        return pred_obs
