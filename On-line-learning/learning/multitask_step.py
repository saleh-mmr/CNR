from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from devices.multiweight_synapse import MultiWeightSynapse
from learning.online_update import OnlineUpdateSpec, apply_online_update


@dataclass(frozen=True)
class TaskSpec:
    name: str
    ap_index: int  # which '+' crosspoint is AP for this task


@dataclass
class MultiTaskUpdateSpec:
    """
    Implements the paper rule:
      'each training step must be carried out for all problems
       before going to the next step' :contentReference[oaicite:3]{index=3}
    """
    online_update: OnlineUpdateSpec


def read_q_from_memristors_for_task(
    phi_s: np.ndarray,
    synapses: List[List[MultiWeightSynapse]],
    ap_index: int,
) -> np.ndarray:
    """
    Q(s,a) readout for a specific task selection (ap_index).
    For one-hot phi, picks the active state row.
    """
    s_idx = int(np.argmax(phi_s))
    n_actions = len(synapses[s_idx])
    q = np.zeros(n_actions, dtype=np.float32)
    for a in range(n_actions):
        q[a] = float(synapses[s_idx][a].weight(ap_index=ap_index))
    return q


def multitask_learning_step(
    tasks: List[TaskSpec],
    # For each task, we provide a single "experience" tuple to train on this step:
    # (phi_s, action, reward, phi_s_next, terminated)
    experiences: Dict[str, Tuple[np.ndarray, int, float, np.ndarray, bool]],
    synapses: List[List[MultiWeightSynapse]],
    gamma: float,
    spec: MultiTaskUpdateSpec,
) -> None:
    """
    Performs ONE paper-style learning step across ALL tasks before returning.

    For each task:
      1) compute delta using memristor-read Q
      2) direction = sign(delta)
      3) update only the active weight (state, action) using paper rule:
            + => increment x_task
            - => increment x_B :contentReference[oaicite:4]{index=4}
    """
    for task in tasks:
        phi_s, action, reward, phi_s_next, terminated = experiences[task.name]

        q = read_q_from_memristors_for_task(phi_s, synapses, ap_index=task.ap_index)
        q_sa = float(q[action])

        q_next = read_q_from_memristors_for_task(phi_s_next, synapses, ap_index=task.ap_index)
        y = float(reward) if terminated else float(reward + gamma * np.max(q_next))
        delta = float(y - q_sa)

        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

        active_state = int(np.argmax(phi_s))
        apply_online_update(
            synapse=synapses[active_state][action],
            direction=direction,
            ap_index=task.ap_index,
            spec=spec.online_update,
        )
