from envs.multiagentenv import MultiAgentEnv
from .find_treasure import FindTreasure
import numpy as np

class FindTreasureWrapper(MultiAgentEnv):
    def __init__(self,
                 common_reward,
                 reward_scalarisation,
                 **kwargs):
        self.env = FindTreasure(**kwargs)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env._max_steps

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        # 无实际影响，先设置为False
        truncated = False
        terminated = all(dones)
        return obs, rewards, terminated, truncated, info

    # def get_obs(self):
    #     return self.env.get_obs()  # list

    # def get_obs_agent(self, agent_id):
    #     obs = self.get_obs()
    #     return obs[agent_id]

    # def get_obs_size(self):
    #     return self.env._obs_len

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
        return (self.env._obs_len - (self.env._obs_len // 3 - 1), ((self.n_agents - 1), 2))
    
    def get_state(self):
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def get_state_size(self):
        return self.get_obs_size() * self.n_agents
    
    def get_obs_size(self):
        return int(self.env._obs_len - (self.env._obs_len // 3 - 1) + (self.n_agents - 1) * 2)
    
    def get_obs(self):
        """
        将FT环境的原始观测向量重新格式化。

        对于每个智能体 i，新的观测向量将包含：
        1. 智能体 i 完整的原始观测（包含自身坐标、步数、视野内的环境信息）。
        2. 按ID顺序排列的所有智能体 j 的全局位置属性（2个特征：x, y）。
        3. 如果某个智能体 j 不在 i 的视野内，其位置属性将被置为-1。

        Args:
            raw_obs (List[List[float]]): `self.get_agent_obs()` 的原始输出。

        Returns:
            List[List[float]]: 一个列表，其中每个元素是对应智能体的格式化观测向量（纯列表格式）。
        """
        raw_obs = self.env.get_obs()
        formatted_obs_list = []

        # 为每个智能体构建其专属的格式化观测
        for i in range(self.n_agents):
            agent_i = self.env._agents[i]
            # --- 步骤 1: 构建清理后的自身属性 ---
            # 1a. 先获取基本属性 [坐标x, 坐标y, 步数]
            self_attributes = raw_obs[i][:3]
            # 1b. 遍历原始观测的栅格化部分，只保留金矿和石头的信息
            grid_part = raw_obs[i][3:]
            # 原始栅格中每个单元格有3个特征: (agent_strength, gold_level, stone_level)
            for k in range(0, len(grid_part), 3):
                # 我们跳过第k个元素(agent_strength)，只拼接后续两个
                gold_level = grid_part[k + 1]
                stone_level = grid_part[k + 2]
                self_attributes.extend([gold_level, stone_level])
            # --- 步骤 2: 构建全局社交模块 ---
            all_agents_attributes = []
            for j in range(self.n_agents):
                if i == j:
                    continue
                agent_j = self.env._agents[j]
                # 判断智能体j是否在智能体i的视野内
                is_out_of_view = False
                if i != j:
                    is_out_of_view = self.env._entity_not_in_obsrange(agent_i.pos, agent_j.pos)
                if is_out_of_view:
                    # 如果不在视野内，位置属性置为-1
                    agent_j_pos_attributes = [-1.0, -1.0]
                else:
                    # 如果在视野内，计算其视野内相对坐标
                    agent_j_pos_attributes = raw_obs[j][:2]
                    agent_j_pos_attributes[0] = (agent_j.pos[0] - agent_i.pos[0] + self.env._agent_view[0]) / (self.env._grid_shape[0] - 1)
                    agent_j_pos_attributes[1] = (agent_j.pos[1] - agent_i.pos[1] + self.env._agent_view[1]) / (self.env._grid_shape[1] - 1)
                    assert agent_j_pos_attributes[0] >= 0.0 and agent_j_pos_attributes[1] >= 0.0
                all_agents_attributes.extend(agent_j_pos_attributes)
            # --- 步骤 3: 拼接成最终的观测向量 ---
            final_obs_vector = self_attributes + all_agents_attributes
            formatted_obs_list.append(final_obs_vector)
        
        return formatted_obs_list
