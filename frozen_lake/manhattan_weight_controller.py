import torch

class ManhattanWeightController:
    def __init__(self, model):
        """
        step_size : conductance increment per index step
        """
        self.model = model
        self.step_size = 0.0001

        self.state = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            device = param.device
            shape = param.data.shape

            # index state (source of truth)
            g_plus_idx = torch.ones(shape, dtype=torch.long, device=device)
            g_minus_idx = torch.zeros(shape, dtype=torch.long, device=device)

            # conductance tensors (derived)
            g_plus = torch.empty(shape, dtype=param.dtype, device=device)
            g_minus = torch.empty(shape, dtype=param.dtype, device=device)

            # initialize conductances
            g_plus.copy_(self._conductance(g_plus_idx, param.dtype))
            g_minus.copy_(self._conductance(g_minus_idx, param.dtype))

            self.state[name] = {
                "param": param,
                "g_plus_idx": g_plus_idx,
                "g_minus_idx": g_minus_idx,
                "g_plus": g_plus,
                "g_minus": g_minus,
            }

            # initialize weight
            param.data.copy_(g_plus - g_minus)

    def _conductance(self, idx, dtype):
        conductance = self.step_size * idx.to(dtype)
        return conductance

    @torch.no_grad()
    def step(self):
        for st in self.state.values():
            param = st["param"]
            grad = param.grad
            if grad is None:
                continue

            valid = torch.isfinite(grad)
            pos = (grad > 0) & valid   # want W down  → increase G-
            neg = (grad < 0) & valid   # want W up    → increase G+

            gp_idx = st["g_plus_idx"]
            gm_idx = st["g_minus_idx"]

            # ---- Manhattan updates (index space only) ----
            if pos.any():
                gm_idx[pos] += 1

            if neg.any():
                gp_idx[neg] += 1

            # ---- recompute conductances ----
            st["g_plus"].copy_(self._conductance(gp_idx, param.dtype))
            st["g_minus"].copy_(self._conductance(gm_idx, param.dtype))

            # ---- write weight ----
            param.data.copy_(st["g_plus"] - st["g_minus"])
