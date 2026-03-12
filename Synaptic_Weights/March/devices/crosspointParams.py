class CrossPointParams:
    """
    Parameters from experiments:
      - GP = a*log10(x) + b
      - if G_P <= G_threshold:
            G_AP = G_P
        else:
            G_AP = G_P * (1 + (G_P / G_S)^(3/4))
        - Noise: add Gaussian noise_realization to G
    """

    def __init__(self, a, b, g_s, g_threshold, sigma_pulse_noise):
        self.a = a
        self.b = b
        self.g_s = g_s
        self.g_threshold = g_threshold
        self.sigma_pulse_noise = sigma_pulse_noise

    def get_sigma(self):
        return self.sigma_pulse_noise