from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import gymnasium as gym

@dataclass(frozen=True)
class FrozenLakeAdapterSpec:
    map_name: str = "4x4"
    is_slippery: bool = False
    seed: int = 0
    feature_dim: int = 48  # shared D


class FrozenLakeAdapter:
    """
    FrozenLake: discrete state in [0..15] for 4x4
    We embed into R^48 by placing one-hot into first 16 dims.
    """
    def __init__(self, spec):
        self.spec = spec
        self.env = gym.make("FrozenLake-v1", map_name=spec.map_name, is_slippery=spec.is_slippery)
        self.n_states = int(self.env.observation_space.n)  # 16 for 4x4
        self.n_actions = int(self.env.action_space.n)      # 4

    def reset(self):
        obs, info = self.env.reset(seed=self.spec.seed)
        return self.phi(int(obs)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self.phi(int(obs)), float(reward), bool(terminated), bool(truncated), info

    def phi(self, s):
        x = np.zeros(self.spec.feature_dim, dtype=np.float32)
        x[s] = 1.0
        return x

    def close(self):
        self.env.close()


@dataclass(frozen=True)
class CliffWalkingAdapterSpec:
    seed: int = 0
    feature_dim: int = 48  # CliffWalking has 48 states already


class CliffWalkingAdapter:
    """
    CliffWalking: discrete state in [0..47]
    We embed into R^48 as a one-hot (full length).
    """
    def __init__(self, spec):
        self.spec = spec
        self.env = gym.make("CliffWalking-v1")
        self.n_states = int(self.env.observation_space.n)  # 48
        self.n_actions = int(self.env.action_space.n)      # 4

    def reset(self):
        obs, info = self.env.reset(seed=self.spec.seed)
        return self.phi(int(obs)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self.phi(int(obs)), float(reward), bool(terminated), bool(truncated), info

    def phi(self, s):
        x = np.zeros(self.spec.feature_dim, dtype=np.float32)
        x[s] = 1.0
        return x

    def close(self):
        self.env.close()
