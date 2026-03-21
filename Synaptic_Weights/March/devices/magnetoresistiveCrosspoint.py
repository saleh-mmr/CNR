from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class MagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Positive crosspoint with magnetoresistance.
    """
    def conductance_p(self, state: CrosspointState):
        return self.calculate_conductance_p(state) * 0.000000000001



    def conductance_ap(self, state: CrosspointState):
        g_p = self.calculate_conductance_p(state)
        return g_p
        # g_threshold = self.params.g_threshold
        # if g_p <= g_threshold:
        #     return float(g_p)
        # g_s = self.params.g_s
        # return float(g_p * (1.0 + (g_p / g_s) ** (3.0 / 4.0)))
