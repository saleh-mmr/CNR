from __future__ import annotations
import numpy as np
import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.config as config

class CartPoleEnv:
    def __init__(self, seed):
        self.env = gym.make("CartPole-v1")

        self.n_actions = int(self.env.action_space.n)
        self.obs_dim = int(self.env.observation_space.shape[0])  # should be 4
        assert self.obs_dim == 4
        if seed is None:
            config_seed = config.get_default_seed()
            self._np_rng = np.random.default_rng(config_seed)
        else:
            self._np_rng = np.random.default_rng(seed)

        # Typical scale factors (not exact bounds, but practical normalization)
        self._scale = np.array([4.8, 5.0, 0.418, 5.0], dtype=np.float32)

    def reset(self):
        config_seed = config.get_default_seed()
        obs, info = self.env.reset(seed=config_seed)
        obs = np.asarray(obs, dtype=np.float32)
        return obs, self.phi(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = np.asarray(obs, dtype=np.float32)
        return obs, self.phi(obs), float(reward), bool(terminated), bool(truncated), info

    def phi(self, obs):
        """
        Feature encoder φ(s).
        Minimal version: normalized raw observation (4 features).
        """
        return (obs / self._scale).astype(np.float32)

    def sample_action(self) -> int:
        return int(self.env.action_space.sample())

    def close(self):
        self.env.close()