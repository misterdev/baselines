from flatland.envs.rail_env import RailEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from flatland.envs.observations import TreeObsForRailEnv
from ray.rllib.utils.seed import seed as set_seed
from flatland.envs.generators import complex_rail_generator, random_rail_generator
import numpy as np
from flatland.envs.predictions import DummyPredictorForRailEnv


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
                prediction_builder_object=DummyPredictorForRailEnv())

        if self.predefined_env:
            self.env.load(config['load_env_path'])
                # '/home/guillaume/EPFL/Master_Thesis/flatland/baselines/torch_training/railway/complex_scene.pkl')

        self.width = self.env.width
        self.height = self.env.height


    
    def reset(self):
        self.agents_done = []
        if self.predefined_env:
            obs = self.env.reset(False, False)
        else:
            obs = self.env.reset()

        predictions = self.env.predict()
        pred_pos = np.concatenate([[x[:, 1:3]] for x in list(predictions.values())], axis=0)

        o = dict()

        for i_agent in range(len(self.env.agents)):

            # prediction of collision that will be added to the observation
            # Allows to the agent to know which other train is is about to meet (maybe will come
            # up with a priority order of trains).
            pred_obs = np.zeros((len(predictions[0]), len(self.env.agents)))

            for time_offset in range(len(predictions[0])):

                # We consider a time window of t-1; t+1 to find a collision
                collision_window = list(range(max(time_offset - 1, 0), min(time_offset + 2, len(predictions[0]))))

                coord_agent = pred_pos[i_agent, time_offset, 0] + 1000*pred_pos[i_agent, time_offset, 1]

                # x coordinates of all other train in the time window
                x_coord_other_agents = pred_pos[list(range(i_agent)) + list(range(i_agent+1, len(self.env.agents)))][
                                                :, collision_window, 0]

                # y coordinates of all other train in the time window
                y_coord_other_agents = pred_pos[list(range(i_agent)) + list(range(i_agent + 1, len(self.env.agents)))][
                                                :, collision_window, 1]

                coord_other_agents = x_coord_other_agents + 1000*y_coord_other_agents

                # collision_info here contains the index of the agent colliding with the current agent
                for collision_info in np.argwhere(coord_agent == coord_other_agents)[:, 0]:
                    pred_obs[time_offset, collision_info + 1*(collision_info >= i_agent)] = 1

            agent_id_one_hot = np.zeros(len(self.env.agents))
            agent_id_one_hot[i_agent] = 1
            o[i_agent] = [obs[i_agent], agent_id_one_hot, pred_obs]

        self.old_obs = o
        oo = dict()
        for i_agent in range(len(self.env.agents)):
            oo[i_agent] = [o[i_agent], o[i_agent][0], o[i_agent][1], o[i_agent][2]]
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict
        return oo

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        # print(obs)

        d = dict()
        r = dict()
        o = dict()
        # print(self.agents_done)
        # print(dones)
        predictions = self.env.predict()
        pred_pos = np.concatenate([[x[:, 1:3]] for x in list(predictions.values())], axis=0)

        for i_agent in range(len(self.env.agents)):
            if i_agent not in self.agents_done:
                # prediction of collision that will be added to the observation
                # Allows to the agent to know which other train is is about to meet (maybe will come
                # up with a priority order of trains).
                pred_obs = np.zeros((len(predictions[0]), len(self.env.agents)))

                for time_offset in range(len(predictions[0])):

                    # We consider a time window of t-1; t+1 to find a collision
                    collision_window = list(range(max(time_offset - 1, 0), min(time_offset + 2, len(predictions[0]))))

                    coord_agent = pred_pos[i_agent, time_offset, 0] + 1000*pred_pos[i_agent, time_offset, 1]

                    # x coordinates of all other train in the time window
                    x_coord_other_agents = pred_pos[list(range(i_agent)) + list(range(i_agent+1, len(self.env.agents)))][
                                                    :, collision_window, 0]

                    # y coordinates of all other train in the time window
                    y_coord_other_agents = pred_pos[list(range(i_agent)) + list(range(i_agent + 1, len(self.env.agents)))][
                                                    :, collision_window, 1]

                    coord_other_agents = x_coord_other_agents + 1000*y_coord_other_agents

                    # collision_info here contains the index of the agent colliding with the current agent
                    for collision_info in np.argwhere(coord_agent == coord_other_agents)[:, 0]:
                        pred_obs[time_offset, collision_info + 1*(collision_info >= i_agent)] = 1

                agent_id_one_hot = np.zeros(len(self.env.agents))
                agent_id_one_hot[i_agent] = 1
                o[i_agent] = [obs[i_agent], agent_id_one_hot, pred_obs]
                r[i_agent] = rewards[i_agent]
                d[i_agent] = dones[i_agent]

        d['__all__'] = dones['__all__']

        # for agent, done in dones.items():
        #     if agent not in self.agents_done:
        #         if agent != '__all__':
        # #            o[agent] = obs[agent]
        #             #one_hot_agent_encoding = np.zeros(len(self.env.agents))
        #             #one_hot_agent_encoding[agent] += 1
        #             o[agent] = obs[agent]#np.append(obs[agent], one_hot_agent_encoding)
        #
        #
        #         d[agent] = dones[agent]

        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict
        #print('Old OBS #####', self.old_obs)
        oo = dict()
        for i_agent in range(len(self.env.agents)):
            if i_agent not in self.agents_done:
                oo[i_agent] = [o[i_agent], self.old_obs[i_agent][0], self.old_obs[i_agent][1],
                            self.old_obs[i_agent][2]]
        
        self.old_obs = o
        for agent, done in dones.items():
            if done and agent != '__all__':
                self.agents_done.append(agent)

        #print(obs)
        #return obs, rewards, dones, infos
        # oo = dict()
        # oo['agents'] = o
        # o['global'] = np.ones((17, 17)) * 17

        # o[0] = [o[0], np.ones((17, 17)) * 17]
        # o['global_obs'] = np.ones((17, 17)) * 17
        # r['global_obs'] = 0
        # d['global_obs'] = True
        return oo, r, d, infos

    def get_agent_handles(self):
        return self.env.get_agent_handles()

    def get_num_agents(self):
        return self.env.get_num_agents()
