import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.seed import seed as set_seed

from flatland.envs.generators import complex_rail_generator, random_rail_generator
from flatland.envs.rail_env import RailEnv


class RailEnvRLLibWrapper(MultiAgentEnv):

    def __init__(self, config):

        super(MultiAgentEnv, self).__init__()

        # Environment ID if num_envs_per_worker > 1
        if hasattr(config, "vector_index"):
            vector_index = config.vector_index
        else:
            vector_index = 1

        self.predefined_env = False

        if config['rail_generator'] == "complex_rail_generator":
            self.rail_generator = complex_rail_generator(nr_start_goal=config['number_of_agents'],
                                                         min_dist=config['min_dist'],
                                                         nr_extra=config['nr_extra'],
                                                         seed=config['seed'] * (1 + vector_index))

        elif config['rail_generator'] == "random_rail_generator":
            self.rail_generator = random_rail_generator()
        elif config['rail_generator'] == "load_env":
            self.predefined_env = True
            self.rail_generator = random_rail_generator()
        else:
            raise (ValueError, f'Unknown rail generator: {config["rail_generator"]}')

        set_seed(config['seed'] * (1 + vector_index))
        self.env = RailEnv(width=config["width"], height=config["height"],
                           number_of_agents=config["number_of_agents"],
                           obs_builder_object=config['obs_builder'], rail_generator=self.rail_generator)

        if self.predefined_env:
            self.env.load_resource('torch_training.railway', 'complex_scene.pkl')

        self.width = self.env.width
        self.height = self.env.height
        self.step_memory = config["step_memory"]

        # needed for the renderer
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict

    def reset(self):
        self.agents_done = []
        if self.predefined_env:
            obs = self.env.reset(False, False)
        else:
            obs = self.env.reset()

        # RLLib only receives observation of agents that are not done.
        o = dict()

        for i_agent in range(len(self.env.agents)):
            data, distance, agent_data = self.env.obs_builder.split_tree(tree=np.array(obs[i_agent]),
                                                                         current_depth=0)
            o[i_agent] = [data, distance, agent_data]

        # needed for the renderer
        self.rail = self.env.rail
        self.agents = self.env.agents
        self.agents_static = self.env.agents_static
        self.dev_obs_dict = self.env.dev_obs_dict

        # If step_memory > 1, we need to concatenate it the observations in memory, only works for
        # step_memory = 1 or 2 for the moment
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

        for i_agent in range(len(self.env.agents)):
            if i_agent not in self.agents_done:
                data, distance, agent_data = self.env.obs_builder.split_tree(tree=np.array(obs[i_agent]),
                                                                             current_depth=0)

                o[i_agent] = [data, distance, agent_data]
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