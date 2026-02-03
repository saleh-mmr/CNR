from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class PulseLogEntry:
    episode: int
    step: int
    x_plus: int
    x_bias: int
    weight: float
    reward: float


@dataclass
class PulseLogger:
    """
    Logs pulse indices and resulting weight values.
    This is NOT an RL logger — this is a PHYSICS logger.
    """
    entries: List[PulseLogEntry] = field(default_factory=list)

    def log(
        self,
        episode: int,
        step: int,
        x_plus: int,
        x_bias: int,
        weight: float,
        reward: float,
    ) -> None:
        self.entries.append(
            PulseLogEntry(
                episode=episode,
                step=step,
                x_plus=x_plus,
                x_bias=x_bias,
                weight=weight,
                reward=reward,
            )
        )

    def to_numpy(self):
        """
        Convert logs to NumPy arrays for plotting/analysis.
        """
        return {
            "episode": np.array([e.episode for e in self.entries]),
            "step": np.array([e.step for e in self.entries]),
            "x_plus": np.array([e.x_plus for e in self.entries]),
            "x_bias": np.array([e.x_bias for e in self.entries]),
            "weight": np.array([e.weight for e in self.entries]),
            "reward": np.array([e.reward for e in self.entries]),
        }
