from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class TwoTaskLogEntry:
    global_step: int
    w_fl: float
    w_cl: float
    episodes_fl: int
    episodes_cl: int


@dataclass
class TwoTaskLogger:
    entries: List[TwoTaskLogEntry] = field(default_factory=list)

    def log(
        self,
        global_step: int,
        w_fl: float,
        w_cl: float,
        episodes_fl: int,
        episodes_cl: int,
    ):
        self.entries.append(
            TwoTaskLogEntry(
                global_step=global_step,
                w_fl=w_fl,
                w_cl=w_cl,
                episodes_fl=episodes_fl,
                episodes_cl=episodes_cl,
            )
        )

    def to_numpy(self):
        return {
            "step": np.array([e.global_step for e in self.entries]),
            "w_fl": np.array([e.w_fl for e in self.entries]),
            "w_cl": np.array([e.w_cl for e in self.entries]),
            "ep_fl": np.array([e.episodes_fl for e in self.entries]),
            "ep_cl": np.array([e.episodes_cl for e in self.entries]),
        }
