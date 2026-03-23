from .crosspointParams import CrossPointParams
from .crosspointState import CrosspointState
import numpy as np


class BaseCrosspoint:
    def __init__(self, params: CrossPointParams, state: CrosspointState):
        self.params = params
        self.state = state

    #  Following Method Redraws Noise
    def redraw_noise(self, sigma):
        noise = float(np.random.normal(0.0, sigma)) if sigma > 0 else 0.0
        return noise

    # Following Method Changes State (increment X and redraw noise)
    def update_state(self):
        self.state.increment_index()
        sigma = self.params.get_sigma()
        noise = self.redraw_noise(sigma)
        self.state.update_noise(noise)

    # Following Method Compute noisy G_p for Current State **DOES NOT UPDATE STATE**
    def calculate_conductance_p(self, state):
        index = state.get_state()
        a = float(self.params.a)
        b = float(self.params.b)
        conductance_without_noise = float(a * np.log10(index) + b)
        return float(conductance_without_noise)
