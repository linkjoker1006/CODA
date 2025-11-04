from envs.multiagentenv import MultiAgentEnv
from .predator_prey import PredatorPrey
import numpy as np

class PredatorPreyWrapper(MultiAgentEnv):
    def __init__(self,
                 common_reward,
                 reward_scalarisation,
                 **kwargs):
        self.env = PredatorPrey(**kwargs)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env._max_steps

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        # 无实际影响，先设置为False
        truncated = False
        terminated = all(dones)
        return obs, rewards, terminated, truncated, info

    # def get_obs_size(self):
    #     return self.env._obs_len
    
    # def get_obs(self):
    #     return self.env.get_obs()  # list

    # def get_obs_agent(self, agent_id):
    #     obs = self.get_obs()
    #     return obs[agent_id]

    # def get_state(self):
    #     obs = self.get_obs()
    #     return np.array(obs).flatten()

    # def get_state_size(self):
    #     return self.get_obs_size() * self.n_agents
    
    # def format_obs_shape(self):
    #     # 自身属性, 教师数量, 教师属性维度
    #     return (self.env._obs_len, ((self.n_agents - 1), 0))

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        # 所有动作都可用
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        return self.env.action_space[0].n

    def reset(self):
        obs = self.env.reset()
        return obs, {}

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def save_replay(self):
        return self.env.render('rgb_array')

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info["state_shape"] = self.get_state_size()
        env_info["obs_shape"] = self.get_obs_size()
        env_info["obs_component"] = self.format_obs_shape()
        print(env_info)
        return env_info

    def get_stats(self):
        return {}
    
    def format_obs_shape(self):
        # 自身属性, 教师数量, 教师属性维度
        return (2 + self.env._agent_view_mask[0] * self.env._agent_view_mask[1] + 1, ((self.n_agents - 1), 2))
    
    def get_state(self):
        obs = self.get_obs()
        return np.array(obs).flatten().tolist()
    
    def get_state_size(self):
        return self.get_obs_size() * self.n_agents
    
    def get_obs_size(self):
        return int(2 + self.env._agent_view_mask[0] * self.env._agent_view_mask[1] + 1 + (self.n_agents - 1) * 2)
    
    def get_obs(self):
        formatted_obs_list = []
        origin_obs = self.env.get_agent_obs()
        _agent_view = [self.env._agent_view_mask[0] // 2, self.env._agent_view_mask[1] // 2]
        for agent_i in range(self.n_agents):
            pos = self.env.agent_pos[agent_i]
            _agent_i_obs = origin_obs[agent_i]
            
            all_agents_attributes = []
            for agent_j in range(self.n_agents):
                if agent_i == agent_j:
                    continue
                agent_j_pos_attributes = [-1, -1]
                if not self.env._entity_not_in_obsrange(pos, self.env.agent_pos[agent_j]):
                    # 如果在视野内，计算其视野内相对坐标
                    agent_j_pos_attributes[0] = (self.env.agent_pos[agent_j][0] - pos[0] + _agent_view[0]) / (self.env._grid_shape[0] - 1)
                    agent_j_pos_attributes[1] = (self.env.agent_pos[agent_j][1] - pos[1] + _agent_view[1]) / (self.env._grid_shape[1] - 1)
                    assert agent_j_pos_attributes[0] >= 0.0 and agent_j_pos_attributes[1] >= 0.0
                all_agents_attributes.extend(agent_j_pos_attributes)
            _agent_i_obs += all_agents_attributes
            
            formatted_obs_list.append(_agent_i_obs)
        
        return formatted_obs_list
