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
from .official.wrapper import StarCraftCapabilityEnvWrapper


class StarCraft2Env2Wrapper(StarCraftCapabilityEnvWrapper):
    
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

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

        nf_al = self.env.get_ally_num_attributes()
        nf_en = self.env.get_enemy_num_attributes()

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
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.env.n_agents,
            "n_allies": self.env.n_agents - 1,
            "n_enemies": self.env.n_enemies,
            "episode_limit": self.env.episode_limit,

            "n_normal_actions": self.env.n_actions_no_attack,
            "state_ally_feats_size": self.env.get_ally_num_attributes(),
            "state_enemy_feats_size": self.env.get_enemy_num_attributes(),
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
