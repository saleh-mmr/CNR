from base_crosspoint import BaseCrosspoint
from crosspoint_state import CrosspointState

class NonMagnetoresistiveCrosspoint(BaseCrosspoint):
    """
    Negative crosspoint.
    """
    def conductance_p(self, state: CrosspointState) -> float:
        return self.gp(state)