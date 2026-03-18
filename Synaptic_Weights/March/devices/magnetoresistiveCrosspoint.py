import numpy as np
from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """

    def conductance_p(self, state: CrosspointState) -> float:
        # return self.calculate_conductance_p(state)
        # index = state.get_state()
        # a = float(self.params.a)
        # b = float(self.params.b)
        # conductance_without_noise = float(a * np.log10(index) + b)
        return float(0.0)

    def conductance_ap(self, state: CrosspointState) -> float:
        # g_p = self.calculate_conductance_p(state)
        # g_threshold = self.params.g_threshold
        # if g_p <= g_threshold:
        #     return float(g_p)
        # g_s = self.params.g_s
        # return float(g_p * (1.0 + (g_p / g_s) ** (3.0 / 4.0)))
        index = 2 if state.get_state() == 1 else state.get_state()
        a = float(self.params.a)
        b = float(self.params.b)
        conductance_without_noise = float(a * np.log10(index) + b)
        return float(conductance_without_noise)
