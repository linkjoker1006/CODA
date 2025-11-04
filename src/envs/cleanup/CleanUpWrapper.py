from envs.multiagentenv import MultiAgentEnv
from .ssd import SSD
import numpy as np

class CleanUpWrapper(MultiAgentEnv):
    def __init__(self,
                 common_reward,
                 reward_scalarisation,
                 **kwargs):
        self.env = SSD(**kwargs)
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
        pass

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
        return (2 + (self.env.dim_obs[0] * self.env.dim_obs[1] - 1) * 5, ((self.n_agents - 1), 2))
    
    def get_state(self):
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def get_state_size(self):
        return self.get_obs_size() * self.n_agents
    
    def get_obs_size(self):
        return int(2 + (self.env.dim_obs[0] * self.env.dim_obs[1] - 1) * 5 + (self.n_agents - 1) * 2)
    
    def get_obs(self):
        origin_obs = self.env.env.get_agent_obs()
        teacher_obs = []
        new_obs = []
        for i in range(self.n_agents):
            # 先添加自身位置
            agent_id = 'agent-' + str(i)
            agent_i = self.env.env.agents[agent_id]
            self_attributes = [agent_i.pos[0] - self.env.view_size, agent_i.pos[1] - self.env.view_size]
            agent_obs = origin_obs[i]
            for j in range(0, agent_obs.shape[0], 3):
                if j // 3 == agent_obs.shape[0] // 3 // 2:
                    # 自己位于中心位置
                    continue
                # 河水、垃圾、苹果、清洁光束、墙壁
                water_level = 0
                waste_level = 0
                apple_level = 0
                clean_level = 0
                wall_level = 0
                if agent_obs[j] == 159 and agent_obs[j + 1] == 67 and agent_obs[j + 2] == 255:
                    pass
                elif agent_obs[j] == 2 and agent_obs[j + 1] == 81 and agent_obs[j + 2] == 154:
                    pass
                elif agent_obs[j] == 204 and agent_obs[j + 1] == 0 and agent_obs[j + 2] == 204:
                    pass
                elif agent_obs[j] == 216 and agent_obs[j + 1] == 30 and agent_obs[j + 2] == 54:
                    pass
                elif agent_obs[j] == 99 and agent_obs[j + 1] == 156 and agent_obs[j + 2] == 194:
                    water_level = 1
                elif agent_obs[j] == 113 and agent_obs[j + 1] == 75 and agent_obs[j + 2] == 24:
                    waste_level = 1
                elif agent_obs[j] == 0 and agent_obs[j + 1] == 255 and agent_obs[j + 2] == 0:
                    apple_level = 1
                elif agent_obs[j] == 100 and agent_obs[j + 1] == 255 and agent_obs[j + 2] == 255:
                    clean_level = 1
                elif agent_obs[j] == 180 and agent_obs[j + 1] == 180 and agent_obs[j + 2] == 180:
                    wall_level = 1
                elif agent_obs[j] == 0 and agent_obs[j + 1] == 0 and agent_obs[j + 2] == 0:
                    # 空地用00000表示
                    pass
                else:
                    assert False
                self_attributes.extend([water_level, waste_level, apple_level, clean_level, wall_level])
            all_agents_attributes = []
            for k in range(self.n_agents):
                if k == i:
                    continue
                agent_k = self.env.env.agents['agent-' + str(k)]
                if self.env._entity_not_in_obsrange(agent_i.pos, agent_k.pos):
                    agent_k_pos_attributes = [-1.0, -1.0]
                else:
                    agent_k_pos_attributes = [agent_k.pos[0] - agent_i.pos[0] + self.env.view_size, agent_k.pos[1] - agent_i.pos[1] + self.env.view_size]
                    assert agent_k_pos_attributes[0] >= 0.0 and agent_k_pos_attributes[1] >= 0.0
                all_agents_attributes.extend(agent_k_pos_attributes)
            new_obs.append(self_attributes + all_agents_attributes)

        return new_obs
