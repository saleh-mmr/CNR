import numpy as np

from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Negative crosspoint.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        # return self.calculate_conductance_p(state)
        # scale = 0.006
        # return float(state.get_state() * scale)
        scale = 24
        index = state.get_state()
        return float(scale * np.log10(index))