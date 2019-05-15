from flatland.envs.rail_env import RailEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from flatland.core.env_observation_builder import TreeObsForRailEnv
from flatland.envs.generators import random_rail_generator


class RailEnvRLLibWrapper(MultiAgentEnv):

    def __init__(self, config):
                 # width,
                 # height,
                 # rail_generator=random_rail_generator(),
                 # number_of_agents=1,
                 # obs_builder_object=TreeObsForRailEnv(max_depth=2)):
        super(MultiAgentEnv, self).__init__()

        self.env = RailEnv(width=config["width"], height=config["height"], rail_generator=config["rail_generator"],
                number_of_agents=config["number_of_agents"])

    def reset(self):
        self.agents_done = []
        return self.env.reset()

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
        
        #print(obs)
        #return obs, rewards, dones, infos
        return o, r, d, infos
    
    def get_agent_handles(self):
        return self.env.get_agent_handles()
