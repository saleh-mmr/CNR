from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # fallback


@dataclass
class FrozenLakeSpec:
    map_name: str = "4x4"          # "4x4" or "8x8"
    is_slippery: bool = False
    seed: int = 0


class FrozenLakeEnv:
    """
    Thin wrapper around Gym/Gymnasium FrozenLake that also provides a one-hot
    feature encoder φ(s) for linear Q approximators.

    We start here because all later memristor-weight code will plug into:
        Q(s,a) = φ(s)^T θ[:,a]
    """
    def __init__(self, spec: FrozenLakeSpec):
        self.spec = spec
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=spec.map_name,
            is_slippery=spec.is_slippery,
        )
        # FrozenLake observation is discrete: 0..(S-1)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

        # Reproducibility
        self._np_rng = np.random.default_rng(spec.seed)

    def reset(self) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=self.spec.seed)
        s = int(obs)
        return s, self.one_hot(s), info

    def step(self, action: int) -> Tuple[int, np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        s2 = int(obs)
        return s2, self.one_hot(s2), float(reward), bool(terminated), bool(truncated), info

    def one_hot(self, s: int) -> np.ndarray:
        x = np.zeros(self.n_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def sample_action(self) -> int:
        # epsilon-greedy will live elsewhere; this is just a helper.
        return int(self.env.action_space.sample())

    def close(self) -> None:
        self.env.close()
