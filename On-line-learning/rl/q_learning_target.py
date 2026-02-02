from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QLearningSpec:
    gamma: float = 0.99


def q_values_linear(phi_s, theta):
    """
    Linear Q function:
        Q(s, a) = phi(s)^T theta[:, a]

    phi_s: shape (S,)
    theta: shape (S, A)
    returns: shape (A,)
    """
    # (S,) @ (S,A) -> (A,)
    return phi_s @ theta


def td_target_q_learning(
    r,
    phi_s_next,
    theta,
    gamma,
    terminated,
):
    """
    Standard Q-learning target:
        y = noise_realization + gamma * max_a' Q(s',a')   if not terminal
        y = noise_realization                             if terminal
    """
    if terminated:
        return float(r)
    q_next = q_values_linear(phi_s_next, theta)
    return float(r + gamma * np.max(q_next))


def td_error_and_action(
    phi_s,
    action,
    r,
    phi_s_next,
    theta,
    spec,
    terminated,
):
    """
    Returns:
      (delta, q_sa)
    where:
      q_sa  = Q(s,a)
      delta = y - Q(s,a)
    """
    q = q_values_linear(phi_s, theta)
    q_sa = float(q[action])
    y = td_target_q_learning(r, phi_s_next, theta, spec.gamma, terminated)
    delta = float(y - q_sa)
    return delta, q_sa


def weight_update_directions(
    phi_s,
    action,
    delta,
):
    """
    Convert TD error into a per-parameter direction matrix for theta (S,A).

    For linear Q-learning with one-hot features:
        theta[s,a] should increase if delta > 0
        theta[s,a] should decrease if delta < 0
    Only the active state row and chosen action column are involved.

    Output: dir_matrix with values in {-1, 0, +1} shaped (S, A)
      +1 => "increase this weight"
      -1 => "decrease this weight"
       0 => no change
    """
    S = phi_s.shape[0]
    # action dimension inferred later when you allocate theta;
    # here we just create directions for the selected action.
    # We'll return a vector for the active state for simplicity.
    dir_vec = np.zeros(S, dtype=np.int8)

    if delta > 0:
        dir_vec = (phi_s > 0).astype(np.int8)  # +1 at active state
    elif delta < 0:
        dir_vec = -(phi_s > 0).astype(np.int8)  # -1 at active state
    # if delta == 0 -> all zeros

    return dir_vec  # directions for theta[:, action]
