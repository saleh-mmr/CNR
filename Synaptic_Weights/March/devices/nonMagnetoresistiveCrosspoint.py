import numpy as np
from .baseCrosspoint import BaseCrosspoint
from .crosspointState import CrosspointState

class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Negative crosspoint.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        # return self.calculate_conductance_p(state)
        index = state.get_state()
        a = float(self.params.a)
        b = float(self.params.b)
        conductance_without_noise = float(a * np.log10(index) + b)
        return float(conductance_without_noise)