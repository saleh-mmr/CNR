class CrosspointState:
    def __init__(self):
        self.x = 0
        self.noise_realization = 0.0

    def update_noise(self, noise):
        self.noise_realization = noise

    def increment_index(self):
        self.x += 1
        return self.x

    def get_state(self):
        return self.x, self.noise_realization