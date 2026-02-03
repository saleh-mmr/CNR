from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np


@dataclass
class EpisodeRewardEntry:
    episode_index: int
    reward: float


@dataclass
class EpisodeRewardLogger:
    """
    Logs total reward per completed episode for one task.
    """
    entries: List[EpisodeRewardEntry] = field(default_factory=list)

    def log(self, episode_index, reward):
        self.entries.append(EpisodeRewardEntry(episode_index, reward))

    def to_numpy(self):
        return {
            "episode": np.array([e.episode_index for e in self.entries]),
            "reward": np.array([e.reward for e in self.entries]),
        }
