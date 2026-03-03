import numpy as np
import torch
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


network = nn.Sequential(
            nn.Linear(4, 10),
            nn.LeakyReLU(negative_slope=0.01),          # LeakyReLU activation function helps learn non-linear patterns.

            nn.Linear(10, 10),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(10, 2)        # [Q_left, Q_right]  → choose max action
        )
values = np.arange(start=0, stop=10000, step=0.001)
values = torch.from_numpy(values)

state = {}


for name, param in network.named_parameters():
    if not param.requires_grad:
        continue

    device = param.device
    shape = param.data.shape
    print(f"Parameter: {name}, Shape: {shape}")
    g_plus_idx = torch.ones(shape, dtype=torch.long, device=device)  # start at 1
    g_minus_idx = torch.zeros(shape, dtype=torch.long, device=device)  # start at 0
    print(f"g_plus_idx: {g_plus_idx}")
    print(f"g_minus_idx: {g_minus_idx}")

    g_plus = torch.full(shape, values[1], dtype=param.dtype, device=device)
    g_minus = torch.full(shape, values[0], dtype=param.dtype, device=device)
    print(f"g_plus: {g_plus}")
    print(f"g_minus: {g_minus}")

    state[name] = {
        "param": param,
        "g_plus_idx": g_plus_idx,
        "g_minus_idx": g_minus_idx,
        "g_plus": g_plus,
        "g_minus": g_minus,
    }

    print(f"State for {name}: {state[name]}")
    print("-----------------------------")


