import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class Env(object):
    """
    对于环境的封装
    """

    def __init__(self, Edge_IDs, edge_index_map, shortest_paths, num_edge):
        self.env = EnvCore(Edge_IDs, edge_index_map, shortest_paths, num_edge)
        self.num_agent = self.env.num_edge

        self.mappo_obs_dim = self.env.obs1_dim
        self.mappo_action_dim = self.env.action1_dim
        self.mappo_share_obs_dim = self.env.share_obs1_dim
        self.maddpg_obs_dim = self.env.obs2_dim
        self.maddpg_action_dim = self.env.action2_dim
        self.maddpg_share_obs_dim = self.env.share_obs2_dim

        # configure spaces
        self.observation1_space = []
        self.action1_space = []
        self.share_observation1_space = []
        self.observation2_space = []
        self.action2_space = []
        self.share_observation2_space = []

        for agent_idx in range(self.num_agent):
            self.action1_space.append(spaces.Discrete(self.mappo_action_dim))
            self.observation1_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_obs_dim,), dtype=np.float32))
            self.share_observation1_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.mappo_share_obs_dim,), dtype=np.float32))
            self.action2_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.maddpg_action_dim,), dtype=np.float32))
            self.observation2_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.maddpg_obs_dim,), dtype=np.float32))
            self.share_observation2_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.maddpg_share_obs_dim,), dtype=np.float32))

    def step(self, off_actions, allo_actions, edge_vehicle_map, next_edge_vehicle_map, future_trajectory):
        all_task_num, edge_task_nums, edge_comp_task_nums, obs, share_obs, next_obs, next_share_obs, rewards, info, done = self.env.step(
            off_actions, allo_actions, edge_vehicle_map, next_edge_vehicle_map, future_trajectory)
        return all_task_num, edge_task_nums, edge_comp_task_nums, obs, share_obs, next_obs, next_share_obs, rewards, info, done

    def reset(self, edge_vehicle_map, future_trajectory):
        all_task_num, edge_task_nums, obs, share_obs = self.env.reset(edge_vehicle_map, future_trajectory)
        return all_task_num, edge_task_nums, obs, share_obs

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass
