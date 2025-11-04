"""Wrapper around Sequential Social Dilemma environment."""

from .import maps
from .cleanup import CleanupEnv
import numpy as np
from gym import spaces


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True

class SSD(object):
    """Sequential Social Dilemma"""
    def __init__(self, name, obs_height, obs_width, cleaning_penalty, disable_left_right_action, disable_rotation_action, obs_cleaned_1hot, n_agents, map_name, shuffle_spawn, global_ref_point, view_size, random_orientation, beam_width, reward_value, cleanup_params, max_steps, seed):

        self.name = name
        self.dim_obs = [obs_height, obs_width, 3]
        self._obs_len = obs_height * obs_width * 3
        self._max_steps = max_steps
        self.view_size = view_size

        # Original space (not necessarily in this order, see
        # the original ssd files):
        # no-op, up, down, left, right, turn-ccw, turn-cw, penalty, clean
        if (disable_left_right_action and disable_rotation_action):
            self.l_action = 4
            self.cleaning_action_idx = 3
            # up, down, no-op, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 8}
        elif disable_left_right_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # up, down, no-op, rotate cw, rotate ccw, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
        elif disable_rotation_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # left, right, up, down, no-op, clean
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8}
        else:  # full action space except penalty beam
            self.l_action = 8
            self.cleaning_action_idx = 7
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8}

        self.obs_cleaned_1hot = obs_cleaned_1hot

        self.n_agents = n_agents

        if map_name == 'cleanup_small_sym':
            ascii_map = maps.CLEANUP_SMALL_SYM
        elif map_name == 'cleanup_10x10_sym':
            ascii_map = maps.CLEANUP_10x10_SYM
        elif map_name == 'cleanup_large_sym':
            ascii_map = maps.CLEANUP_LARGE_SYM

        self.env = CleanupEnv(ascii_map=ascii_map,
                              num_agents=self.n_agents, render=False,
                              shuffle_spawn=shuffle_spawn,
                              global_ref_point=global_ref_point,
                              view_size=view_size,
                              random_orientation=random_orientation,
                              cleanup_params=cleanup_params,
                              beam_width=beam_width,
                              reward_value=reward_value,
                              cleaning_penalty=cleaning_penalty)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.l_action)] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([self.env.observation_space] * self.n_agents)
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": []}
        self.seed(seed)

        # length of action input to learned reward function
        if self.obs_cleaned_1hot:
            self.l_action_for_r = 2
        else:
            self.l_action_for_r = self.l_action

        self.steps = 0

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.l_action
        env_info["n_agents"] = self.n_agents
        env_info["state_shape"] = self.dim_obs[0] * self.dim_obs[1] * self.dim_obs[2] * self.n_agents
        env_info["obs_shape"] = self.dim_obs[0] * self.dim_obs[1] * self.dim_obs[2]
        env_info["episode_limit"] = self._max_steps
        return env_info

    def get_obs(self):
        return self.process_obs(self.env.get_agent_obs())

    def get_state(self):
        _all_obs = self.process_obs(self.env.get_agent_obs())
        state = np.array(_all_obs).flatten()
        return state

    def get_avail_agent_actions(self, i):
        return [1] * (self.l_action)

    def process_obs(self, obs_list):
        return [(obs / 256.0).tolist() for obs in obs_list]

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self):
        """Resets the environemnt.
        Returns:
            List of agent observations
        """
        obs = self.env.reset()
        self.steps = 0

        return self.process_obs(obs)

    def step(self, actions):
        """Takes a step in env.

        Args:
            actions: list of integers
        Returns:
            List of observations, list of rewards, done, info
        """
        actions = [self.map_to_orig[a] for a in actions]
        actions_dict = {'agent-%d' % idx: actions[idx]
                        for idx in range(self.n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)  # rewards是每个智能体的奖励
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        rewards = list(rewards.values())

        if self.steps == self._max_steps:
            done = [True] * self.n_agents
        else:
            done = list(dones.values())[0: -1]
        rewards = [sum(rewards)] * self.n_agents

        return obs_next, rewards, done, info

    def render(self, mode):
        return self.env.render()

    def rateofwaste(self):
        rateofwaste = self.env.compute_amountofwaste()
        if rateofwaste == 1:
            rateofwaste = 0.99
        return rateofwaste
    
    def _entity_not_in_obsrange(self, pos_i, pos_j):
        if abs(pos_i[0] - pos_j[0]) <= self.view_size and abs(pos_i[1] - pos_j[1]) <= self.view_size:
            return False
        else:
            return True