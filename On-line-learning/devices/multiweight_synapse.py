from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from devices.magnetoresistance import (
    MagnetoresistanceParams,
    MagnetoresistiveCrosspoint,
    NonMagneticCrosspoint,
    CrosspointState,
)


@dataclass
class MultiWeightSynapseSpec:
    """
    Composite synapse spec from the paper.
    """
    n_plus: int                  # N positive crosspoints
    scaling_factor: float        # converts conductance -> ANN weight


class MultiWeightSynapse:
    """
    One composite device (index m in the paper):
      - N '+' crosspoints with magnetoresistance
      - 1 '-' bias crosspoint (no magnetoresistance)
    """
    def __init__(
        self,
        spec: MultiWeightSynapseSpec,
        params: MagnetoresistanceParams,
        rng: Optional[np.random.Generator] = None,
    ):
        self.spec = spec
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()

        # Create crosspoints
        self.plus_devices: List[MagnetoresistiveCrosspoint] = [
            MagnetoresistiveCrosspoint(params=params, rng=self.rng)
            for _ in range(spec.n_plus)
        ]
        self.bias_device = NonMagneticCrosspoint(params=params, rng=self.rng)

        # Create states (pulse index + noise) for each crosspoint
        self.plus_states: List[CrosspointState] = [
            CrosspointState() for _ in range(spec.n_plus)
        ]
        self.bias_state = CrosspointState()

        # Initial noise draw (paper initializes r's)
        for dev, st in zip(self.plus_devices, self.plus_states):
            dev.redraw_noise(st)
        self.bias_device.redraw_noise(self.bias_state)

    # ------------------------------------------------------------------
    # Weight evaluation (paper equations)
    # ------------------------------------------------------------------

    def weight(self, ap_index: int) -> float:
        """
        Compute W_n for the magnetic configuration where plus crosspoint
        `ap_index` is in AP state and all others are in P.

        ap_index: int in [0, N-1]
        """
        assert 0 <= ap_index < self.spec.n_plus

        g_sum = 0.0
        for i, (dev, st) in enumerate(zip(self.plus_devices, self.plus_states)):
            if i == ap_index:
                g_sum += dev.conductance_ap(st)
            else:
                g_sum += dev.conductance_p(st)

        g_bias = self.bias_device.conductance(self.bias_state)

        return float(self.spec.scaling_factor * (g_sum - g_bias))

    # ------------------------------------------------------------------
    # Pulse update primitives (paper on-line rules)
    # ------------------------------------------------------------------

    def increase_plus(self, index: int, n_pulses: int = 1) -> None:
        """
        Paper rule:
          If a weight must be increased, increase the corresponding x index
          and draw a new random noise value r. :contentReference[oaicite:1]{index=1}
        """
        assert 0 <= index < self.spec.n_plus
        self.plus_devices[index].increment_pulses(
            self.plus_states[index], n=n_pulses
        )

    def increase_bias(self, n_pulses: int = 1) -> None:
        """
        Paper rule:
          If a weight must be decreased, increase the bias index x_B
          and draw a new r_B. :contentReference[oaicite:2]{index=2}
        """
        self.bias_device.increment_pulses(self.bias_state, n=n_pulses)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def pulse_indices(self):
        """
        For logging_helper/debugging: return current pulse indices.
        """
        return {
            "plus": [st.x for st in self.plus_states],
            "bias": self.bias_state.x,
        }
