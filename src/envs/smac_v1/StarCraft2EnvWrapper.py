#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：API-Network 
@File    ：StarCraft2EnvWrapper.py
@Author  ：Hao Xiaotian
@Date    ：2022/6/13 16:26 
'''

import copy
import numpy as np
from .official.starcraft2 import StarCraft2Env
from envs.multiagentenv import MultiAgentEnv

class StarCraft2EnvWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit
    
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_stats(self):
        return self.env.get_stats()

    # Add new functions to support permutation operation
    def get_obs_component(self):
        move_feats_dim = self.env.get_obs_move_feats_size()
        enemy_feats_dim = self.env.get_obs_enemy_feats_size()
        ally_feats_dim = self.env.get_obs_ally_feats_size()
        own_feats_dim = self.env.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        return obs_component

    def get_state_component(self):
        if self.env.obs_instead_of_state:
            return [self.env.get_obs_size()] * self.env.n_agents

        nf_al = 4 + self.env.shield_bits_ally + self.env.unit_type_bits
        nf_en = 3 + self.env.shield_bits_enemy + self.env.unit_type_bits

        enemy_state = self.env.n_enemies * nf_en
        ally_state = self.env.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.env.state_last_action:
            size.append(self.env.n_agents * self.env.n_actions)
        if self.env.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self):
        env_info = {
            "state_shape": self.env.get_state_size(),
            "obs_shape": self.env.get_obs_size(),
            "n_actions": self.env.get_total_actions(),
            "n_agents": self.env.n_agents,
            "episode_limit": self.env.episode_limit,

            "n_normal_actions": self.env.n_actions_no_attack,
            "n_enemies": self.env.n_enemies,
            "n_allies": self.env.n_agents - 1,
            "state_ally_feats_size": self.env.get_ally_num_attributes(),  # 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": self.env.get_enemy_num_attributes(),  # 3 + self.shield_bits_enemy + self.unit_type_bits,
            "obs_move_feats_size": self.env.get_obs_move_feats_size(),
            "obs_own_feats_size": self.env.get_obs_own_feats_size(),
            "obs_ally_feats_size": self.env.get_obs_ally_feats_size(),
            "obs_enemy_feats_size": self.env.get_obs_enemy_feats_size(),
            "unit_type_bits": self.env.unit_type_bits,
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.env.map_type,
        }
        print(env_info)
        
        return env_info

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.env.agents.items():
            if self.env.map_type == "MMM" and al_unit.unit_type == self.env.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids
