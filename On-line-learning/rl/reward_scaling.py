from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardScaleSpec:
    scale: float = 1.0   # multiply reward by this
    clip: float = 1.0    # then clip into [-clip, clip]


def scale_and_clip(r, spec):
    v = float(r) * float(spec.scale)
    if v > spec.clip:
        return float(spec.clip)
    if v < -spec.clip:
        return float(-spec.clip)
    return float(v)
