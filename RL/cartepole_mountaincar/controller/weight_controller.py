import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.multiWeightSynapse import MultiWeightSynapse, MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


class ManhattanWeightController:
    def __init__(self, model):
        # Parameters for crosspoints and synapse spec
        a = 1
        b = 1
        c = 1
        g_threshold = 0.8
        sigma_pulse_noise = 0.0
        scaling_factor = 1
        n_problem = 2

        # model is the neural network whose weights we want to control
        self.model = model

        # initialize synapses for each trainable parameter in the model
        self.synapses = []
        self.params = CrossPointParams(a, b, c, g_threshold, sigma_pulse_noise)
        self.spec = MultiWeightSynapseSpec(n_problem, scaling_factor)

        # iterates through all parameters in the model, giving a tuple of (name, parameter tensor)
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            syn_array = np.empty(param.shape, dtype=object)
            # For each element in the parameter tensor, we create a MultiWeightSynapse and add it to the syn_list
            # param.numel() gives the total number of elements in the parameter tensor
            # (e.g., for a parameter tensor of shape (4, 8), param.numel() would be 32)
            for index in np.ndindex(param.shape):
                syn_array[index] = MultiWeightSynapse(self.spec, self.params)

            self.synapses.append((param, syn_array))

    @torch.no_grad()
    def step(self, ap_index=0):
        for param, syn_array in self.synapses:
            if param.grad is None:
                continue

            grad = param.grad

            for index in np.ndindex(param.shape):
                g_value = grad[index].item()
                syn = syn_array[index]
                sign = (g_value > 0) - (g_value < 0)
                if sign == 0:
                    continue
                if sign>0:
                    syn.increase_bias_crosspoint_index()
                elif sign<0:
                    syn.increase_positive_crosspoint_index(ap_index)

                weight = syn.weight(ap_index)
                param[index].copy_(
                    torch.tensor(weight, dtype=param.dtype, device=param.device)
                )