class CrossPointParams:
    """
    Parameters from experiments:
      - GP = g_p_coefficient * log10(index + shift_parameter)
      - GAP = g_ap_coefficient * log10(index)
    """

    def __init__(self, sigma_pulse_noise, g_ap_coefficient, g_p_coefficient, shift_parameter, g_bias_coefficient):
        self.sigma_pulse_noise = sigma_pulse_noise
        self.g_ap_coefficient = g_ap_coefficient
        self.g_p_coefficient = g_p_coefficient
        self.shift_parameter = shift_parameter
        self.g_bias_coefficient = g_bias_coefficient

    def get_sigma(self):
        return self.sigma_pulse_noise