import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        g_p_coefficient = self.params.g_p_coefficient
        index = state.get_state()
        return float(g_p_coefficient * np.log10(index))

    def conductance_ap(self, state: CrosspointState) -> float:
        g_ap_coefficient = self.params.g_ap_coefficient
        index = state.get_state()
        return float(g_ap_coefficient * np.log10(index))
