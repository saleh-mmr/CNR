import torch


class Controller:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def step(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            grad = param.grad
            if grad is None:
                continue

            weight = param.data
            weight += -0.0001 * grad

            param.copy_(weight)