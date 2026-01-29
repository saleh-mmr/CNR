from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MagnetoresistanceParams:
    """
    Parameters from experiments (paper):
      - GP = a*log10(x) + b
      - if GP <= Gthreshold: GAP = GP
        else GAP = GP * (1 + c * (GP - Gthreshold)^(3/4))
      - Noise: add Gaussian r to G (paper writes: G = a*log10(x)+b+r)
    """
    a: float
    b: float
    c: float
    g_threshold: float

    # Noise std (not specified numerically in the paper; you choose it)
    sigma_pulse_noise: float = 0.0

    # Handle log10(0): the paper initializes indices at 0.
    # We implement log10(max(x, 1)) so x=0 behaves like x=1 (baseline).
    min_pulse_index_for_log: int = 1


def gp_from_pulses(x: int, a: float, b: float, min_x: int = 1) -> float:
    """
    Compute G_P from pulse index using paper law:
      G_P = a*log10(x) + b

    Note: paper sets x=0 initially. log10(0) undefined, so we clamp:
      x_eff = max(x, min_x)
    """
    x_eff = max(int(x), int(min_x))
    return float(a * np.log10(x_eff) + b)


def gap_from_gp(gp: float, c: float, g_threshold: float) -> float:
    """
    Paper threshold rule:
      if GP <= Gthreshold:
          GAP = GP
      else:
          GAP = GP * (1 + c*(GP - Gthreshold)^(3/4))
    """
    if gp <= g_threshold:
        return float(gp)
    return float(gp * (1.0 + c * (gp - g_threshold) ** (3.0 / 4.0)))


@dataclass
class CrosspointState:
    """
    Represents one crosspoint's internal programming state for the pulsed-update model.
    Stores:
      - pulse index x
      - current noise sample r (Gaussian); redrawn when x changes (paper algorithm)
    """
    x: int = 0
    r: float = 0.0


class MagnetoresistiveCrosspoint:
    """
    Positive ('+') crosspoint with magnetoresistance.
    It can be evaluated in P or AP configuration.

    Noise handling:
      - We store a noise sample r inside the state.
      - When an update increments x, we redraw r (Gaussian), matching the on-line rule:
          "increase x and draw a new random noise value r" :contentReference[oaicite:1]{index=1}
    """
    def __init__(
        self,
        params: MagnetoresistanceParams,
        rng: Optional[np.random.Generator] = None,
        a_override: Optional[float] = None,
        b_override: Optional[float] = None,
    ):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        # Optional device-to-device variation: different a,b per crosspoint
        self.a = float(a_override) if a_override is not None else float(params.a)
        self.b = float(b_override) if b_override is not None else float(params.b)

    def redraw_noise(self, state: CrosspointState) -> None:
        sigma = float(self.params.sigma_pulse_noise)
        state.r = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0

    def increment_pulses(self, state: CrosspointState, n: int = 1) -> None:
        state.x += int(n)
        # Paper: when x changes, draw a new noise value r
        self.redraw_noise(state)

    def conductance_p(self, state: CrosspointState) -> float:
        gp = gp_from_pulses(
            x=state.x,
            a=self.a,
            b=self.b,
            min_x=self.params.min_pulse_index_for_log,
        )
        return float(gp + state.r)

    def conductance_ap(self, state: CrosspointState) -> float:
        gp_noisy = self.conductance_p(state)  # includes +r as paper writes
        return gap_from_gp(
            gp=gp_noisy,
            c=self.params.c,
            g_threshold=self.params.g_threshold,
        )


class NonMagneticCrosspoint:
    """
    Bias ('-') crosspoint: single-valued (no magnetoresistance).
    Paper still models it with the same GP log law and noise:
      G = a*log10(x)+b+r  (they subtract it in the weight equation). :contentReference[oaicite:2]{index=2}
    """
    def __init__(
        self,
        params: MagnetoresistanceParams,
        rng: Optional[np.random.Generator] = None,
        a_override: Optional[float] = None,
        b_override: Optional[float] = None,
    ):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.a = float(a_override) if a_override is not None else float(params.a)
        self.b = float(b_override) if b_override is not None else float(params.b)

    def redraw_noise(self, state: CrosspointState) -> None:
        sigma = float(self.params.sigma_pulse_noise)
        state.r = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0

    def increment_pulses(self, state: CrosspointState, n: int = 1) -> None:
        state.x += int(n)
        # Paper: redraw noise when x changes
        self.redraw_noise(state)

    def conductance(self, state: CrosspointState) -> float:
        g = gp_from_pulses(
            x=state.x,
            a=self.a,
            b=self.b,
            min_x=self.params.min_pulse_index_for_log,
        )
        return float(g + state.r)
