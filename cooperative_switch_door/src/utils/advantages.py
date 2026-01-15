import torch


def compute_gae_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value,
    gamma: float,
    gae_lambda: float,
):
    """
    Compute team-level GAE advantages and returns for MAPPO.

    Args:
        rewards: [T] float tensor
        dones:   [T] float/bool tensor (1 if episode ended at step t else 0)
        values:  [T] float tensor (V(s_t))
        last_value: scalar (V(s_{T})) for bootstrap, 0 if terminal
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [T]
        returns:    [T]
    """
    assert rewards.dim() == 1
    assert dones.dim() == 1
    assert values.dim() == 1
    T = rewards.shape[0]

    advantages = torch.zeros(T, dtype=torch.float32, device=values.device)
    returns = torch.zeros(T, dtype=torch.float32, device=values.device)

    # make last_value a float tensor on the same device
    if not torch.is_tensor(last_value):
        last_value = torch.tensor(last_value, dtype=torch.float32, device=values.device)
    else:
        last_value = last_value.to(device=values.device, dtype=torch.float32)

    gae = torch.tensor(0.0, dtype=torch.float32, device=values.device)
    next_value = last_value

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return advantages, returns

