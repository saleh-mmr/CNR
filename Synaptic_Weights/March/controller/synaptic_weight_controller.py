import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from devices.multiWeightSynapse import MultiWeightSynapse, MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


class SynapticWeightController:
    def __init__(self, model, ):
        self.model = model

        params = CrossPointParams(a=1.566e-8, b=3.5e-9, g_s=4.32e-7, g_threshold=9e-15, sigma_pulse_noise=0.0)
        spec = MultiWeightSynapseSpec(n_problem=2, scaling_factor=9e7)

        self.synapses = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            shape = param.shape
            if len(shape) == 2:  # weight matrix
                self.synapses[name] = [
                    [MultiWeightSynapse(spec, params) for _ in range(shape[1])]
                    for _ in range(shape[0])
                ]

            elif len(shape) == 1:  # bias vector
                self.synapses[name] = [
                    MultiWeightSynapse(spec, params) for _ in range(shape[0])
                ]


    @torch.no_grad()
    def step(self, ap_index):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid
            neg = (grad < 0) & valid

            if name == "FC.0.weight":
                print(f"Updating indexes for {name} neuron [0,0] (AP index {ap_index}):")
                print(f"Gradient: {grad[0, 0].item():.4f}")
                print()


            if grad.ndim == 2:
                if pos.any():
                    for i in range(pos.shape[0]):
                        for j in range(pos.shape[1]):
                            if pos[i, j]:
                                self.synapses[name][i][j].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        for j in range(neg.shape[1]):
                            if neg[i, j]:
                                self.synapses[name][i][j].increase_positive_crosspoint_index(ap_index)

            elif grad.ndim == 1:
                if pos.any():
                    for i in range(pos.shape[0]):
                        if pos[i]:
                            self.synapses[name][i].increase_bias_crosspoint_index()
                if neg.any():
                    for i in range(neg.shape[0]):
                        if neg[i]:
                            self.synapses[name][i].increase_positive_crosspoint_index(ap_index)


    @torch.no_grad()
    def load_weights(self, ap_index):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            st = self.synapses[name]
            if param.ndim == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        param[i, j].copy_(torch.tensor(st[i][j].weight(ap_index), dtype=param.dtype))
            elif param.ndim == 1:
                for i in range(param.shape[0]):
                    param[i].copy_(torch.tensor(st[i].weight(ap_index), dtype=param.dtype))


            if name == "FC.0.weight":
                print(f"Loaded weights for {name} neuron [0,0] (AP index {ap_index}):")
                print(f"{param[0, 0].item():.4f}")