from crosspointParams import CrossPointParams
from crosspoint_state import CrosspointState
import numpy as np
from Clif_Frozen.utils import config


class BaseCrosspoint:
    def __init__(self, params: CrossPointParams, state: CrosspointState):
        self.params = params
        self.state = state
        self.rng = np.random.default_rng(seed=config.seed)

    #  Following Method Redraws Noise
    def redraw_noise(self, sigma):
        noise = float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0
        return noise


    # Following Method Changes State (increment X and redraw noise)
    def update_state(self):
        self.state.increment_index()
        sigma = self.params.get_sigma()
        noise = self.redraw_noise(sigma)
        self.state.update_noise(noise)

    # Following Method Compute noisy G_p for current State
    def gp(self, state):
        index, state_noise = state.get_state()
        a = float(self.params.a)
        b = float(self.params.b)
        min_x = self.params.min_pulse_index_for_log
        index = max(int(index), int(min_x))
        gp = float(a * np.log10(index) + b)
        return float(gp + state_noise)