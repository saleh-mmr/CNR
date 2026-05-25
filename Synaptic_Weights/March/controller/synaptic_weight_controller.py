import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


class SynapticWeightController:
    def __init__(self, model, g_ap, g_p, shift_parameter, g_bias, noise_stddev):
        self.model = model

        # Conductance parameters
        self.g_ap = float(g_ap)
        self.g_p = float(g_p)
        self.shift_parameter = float(shift_parameter)
        self.g_bias = float(g_bias)
        self.noise_stddev = float(noise_stddev)

        # Same as your old MultiWeightSynapseSpec
        self.n_problem = 3
        self.scaling_factor = 1.0

        # Tensor-based synapse states
        #
        # For each model parameter named `name`, we store:
        #
        # self.bias_x[name]:
        #     bias crosspoint index, same shape as parameter
        #
        # self.bias_noise[name]:
        #     bias crosspoint noise, same shape as parameter
        #
        # self.positive_x[name]:
        #     positive crosspoint index, shape = (n_problem, *parameter.shape)
        #
        # self.positive_noise[name]:
        #     positive crosspoint noise, shape = (n_problem, *parameter.shape)
        self.bias_x = {}
        self.bias_noise = {}
        self.positive_x = {}
        self.positive_noise = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            dtype = param.dtype
            shape = param.shape

            initial_x = self._initial_index(device=device, dtype=dtype)

            self.bias_x[name] = torch.full(
                shape,
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.bias_noise[name] = self._draw_noise(
                shape,
                device=device,
                dtype=dtype,
            )

            self.positive_x[name] = torch.full(
                (self.n_problem, *shape),
                initial_x,
                dtype=dtype,
                device=device,
            )

            self.positive_noise[name] = self._draw_noise(
                (self.n_problem, *shape),
                device=device,
                dtype=dtype,
            )

        # Helps avoid unnecessary reloading later.
        # We keep it simple for now.
        self.current_loaded_ap_index = None
        self.weights_dirty = True

    def _initial_index(self, device, dtype):
        """
        Same logic as CrosspointState.__init__:

            k = g_ap / g_p
            i = 1
            while i**k <= i + shift_parameter:
                i += 1

        Returns the initial index as a float tensor value.
        """
        k = self.g_ap / self.g_p
        i = 1

        while i ** k <= i + self.shift_parameter:
            i += 1

        return float(i)

    def _draw_noise(self, shape, device, dtype):
        """
        Same behavior as np.random.normal(0.0, noise_stddev), but vectorized.

        If noise_stddev <= 0, return zero noise.
        """
        if self.noise_stddev > 0:
            return torch.normal(
                mean=0.0,
                std=self.noise_stddev,
                size=shape,
                device=device,
                dtype=dtype,
            )

        return torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, ap_index):
        """
        Vectorized replacement for the old per-weight update.

        Old behavior:

        if grad > 0:
            increase bias crosspoint index

        if grad < 0:
            increase positive crosspoint index for this ap_index
        """
        assert 0 <= ap_index < self.n_problem

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid

            # Positive gradient:
            # old code called increase_bias_crosspoint_index()
            if pos.any():
                self.bias_x[name][pos] += 1.0
                self.bias_noise[name][pos] = self._draw_noise(
                    self.bias_noise[name][pos].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

            # Negative gradient:
            # old code called increase_positive_crosspoint_index(ap_index)
            if neg.any():
                self.positive_x[name][ap_index][neg] += 1.0
                self.positive_noise[name][ap_index][neg] = self._draw_noise(
                    self.positive_noise[name][ap_index][neg].shape,
                    device=param.device,
                    dtype=param.dtype,
                )

        self.weights_dirty = True

    @torch.no_grad()
    def load_weights(self, ap_index):
        """
        Load physical synaptic weights into the PyTorch model.

        Important:
        This version does NOT check param.grad.
        The old version skipped loading weights when grad was None,
        which could cause incorrect action selection before backprop.
        """
        assert 0 <= ap_index < self.n_problem

        if self.current_loaded_ap_index == ap_index and not self.weights_dirty:
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # ----- Bias conductance -----
            x_bias = self.bias_x[name]
            noise_bias = self.bias_noise[name]

            g_bias = self.g_bias * torch.log10(x_bias) + (1.0 + noise_bias)

            # ----- Positive crosspoint conductance total -----
            g_total = torch.zeros_like(param)

            for problem_index in range(self.n_problem):
                x_pos = self.positive_x[name][problem_index]
                noise_pos = self.positive_noise[name][problem_index]

                if problem_index == ap_index:
                    # AP conductance
                    g = self.g_ap * torch.log10(x_pos) + (1.0 + noise_pos)
                else:
                    # P conductance
                    g = self.g_p * torch.log10(x_pos + self.shift_parameter) + (1.0 + noise_pos)

                g_total += g

            weight = self.scaling_factor * (g_total - g_bias)
            param.copy_(weight)


            if name == "FC.2.weight":
                print(
                    f"load_weights(ap_index={ap_index}) | FC.2.weight[0][0] = "
                    f"{param[0, 0].item():.6f}"
                )

        self.current_loaded_ap_index = ap_index
        self.weights_dirty = False