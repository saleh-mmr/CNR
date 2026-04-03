import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        # return self.calculate_conductance_p(state)
        return 0.0


    def conductance_ap(self, state: CrosspointState) -> float:
        g_p = self.calculate_conductance_p(state)
        g_s = self.params.g_s
        # return float(g_p * (1.0 + (g_p / g_s) ** (3.0 / 4.0)))
        scale = 0.009
        index = state.get_state()
        return float(index * scale)
