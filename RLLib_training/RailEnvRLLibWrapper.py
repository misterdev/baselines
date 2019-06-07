from flatland.envs.rail_env import RailEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.generators import random_rail_generator
from ray.rllib.utils.seed import seed as set_seed
import numpy as np


class RailEnvRLLibWrapper(MultiAgentEnv):

    def __init__(self, config):
                 # width,
                 # height,
                 # rail_generator=random_rail_generator(),
                 # number_of_agents=1,
                 # obs_builder_object=TreeObsForRailEnv(max_depth=2)):
        super(MultiAgentEnv, self).__init__()
        if hasattr(config, "vector_index"):
            vector_index = config.vector_index
        else:
            vector_index = 1
        #self.rail_generator = config["rail_generator"](nr_start_goal=config['number_of_agents'], min_dist=5,
         #                                              nr_extra=30, seed=config['seed'] * (1+vector_index))
        set_seed(config['seed'] * (1+vector_index))
        #self.env = RailEnv(width=config["width"], height=config["height"],
        self.env = RailEnv(width=10, height=20,
                number_of_agents=config["number_of_agents"], obs_builder_object=config['obs_builder'])

        self.env.load('/mount/SDC/flatland/baselines/torch_training/railway/complex_scene.pkl')

        self.width = self.env.width
        self.height = self.env.height


    
    def reset(self):
        self.agents_done = []
        obs = self.env.reset(False, False)
        o = dict()
        # o['agents'] = obs
        # obs[0] = [obs[0], np.ones((17, 17)) * 17]
        # obs['global_obs'] = np.ones((17, 17)) * 17


        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict
        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        # print(obs)

        d = dict()
        r = dict()
        o = dict()
        # print(self.agents_done)
        # print(dones)
        for agent, done in dones.items():
            if agent not in self.agents_done:
                if agent != '__all__':
                    o[agent] = obs[agent]
                    r[agent] = rewards[agent]
    
                d[agent] = dones[agent]

        for agent, done in dones.items():
            if done and agent != '__all__':
                self.agents_done.append(agent)

        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict
        
        #print(obs)
        #return obs, rewards, dones, infos
        # oo = dict()
        # oo['agents'] = o
        # o['global'] = np.ones((17, 17)) * 17

        # o[0] = [o[0], np.ones((17, 17)) * 17]
        # o['global_obs'] = np.ones((17, 17)) * 17
        # r['global_obs'] = 0
        # d['global_obs'] = True
        return o, r, d, infos

    def get_agent_handles(self):
        return self.env.get_agent_handles()

    def get_num_agents(self):
        return self.env.get_num_agents()
