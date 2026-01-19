import numpy as np
import torch
import pandas as pd

class ManhattanWeightController:
    def __init__(self, model, base_scale: float = 9e7):
        """
        model: torch.nn.Module — model whose parameters are controlled
        base_scale: float — scaling factor applied to the generated conductance values
        """
        self.model = model

        # conductance = pd.read_csv(csv_path, header=0)
        # values = conductance.values.astype("float32").reshape(-1)
        # values *= 9e8

        # values = np.arange(start=0, stop=100,step=0.01)

        a = 1.566e-8
        b = 0.350e-8
        sigma = 1.7e-9
        # Allow configuring the base scale for the conductance values
        # `base_scale` passed in parameter overrides the default
        # Keep the numeric behavior identical when using the default value 9e7
        # (original code used 9e7)
        x = np.arange(1, 2000)
        values = (a * np.log10(x) + b).astype(np.float32)
        values *= base_scale
        print(values[:200])
        self.values = torch.from_numpy(values)

        # Cache values tensor per device to avoid repeated .to(device)
        self._values_cache = {}

        self.state = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            shape = param.data.shape

            g_plus_idx = torch.ones(shape, dtype=torch.long, device=device)    # start at 1
            g_minus_idx = torch.zeros(shape, dtype=torch.long, device=device)  # start at 0

            g_plus = torch.full(shape, self.values[1], dtype=param.dtype, device=device)
            g_minus = torch.full(shape, self.values[0], dtype=param.dtype, device=device)

            self.state[name] = {
                "param": param,
                "g_plus_idx": g_plus_idx,
                "g_minus_idx": g_minus_idx,
                "g_plus": g_plus,
                "g_minus": g_minus,
            }

    def _values_on(self, device, dtype):
        key = (device.type, device.index, dtype)
        if key not in self._values_cache:
            self._values_cache[key] = self.values.to(device=device, dtype=dtype)
        return self._values_cache[key]

    # -------------------
    # Step 2: implement step()
    # -------------------
    @torch.no_grad()
    def step(self):
        max_idx = int(self.values.numel() - 1)

        for name, st in self.state.items():
            param = st["param"]
            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid  # want W down
            neg = (grad < 0) & valid  # want W up

            gp = st["g_plus_idx"]
            gm = st["g_minus_idx"]

            # ---- grad > 0 : decrease W ----
            if pos.any():
                can_inc_gm = pos & (gm < max_idx)  # preferred action
                gm[can_inc_gm] += 1  # increase G-

            # ---- grad < 0 : increase W ----
            if neg.any():
                can_inc_gp = neg & (gp < max_idx)  # preferred action
                gp[can_inc_gp] += 1  # increase G+

            # Clamp indices to valid range
            gp.clamp_(0, max_idx)
            gm.clamp_(0, max_idx)

            # Update conductances from indices
            values_dev = self._values_on(param.device, param.dtype)
            st["g_plus"].copy_(values_dev[gp])
            st["g_minus"].copy_(values_dev[gm])

            # Write back weight
            param.data.copy_(st["g_plus"] - st["g_minus"])
