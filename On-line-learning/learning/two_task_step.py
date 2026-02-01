from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from devices.multiweight_synapse import MultiWeightSynapse
from learning.online_update import OnlineUpdateSpec, apply_online_update


@dataclass(frozen=True)
class TaskSpec:
    name: str
    ap_index: int


@dataclass
class TwoTaskUpdateSpec:
    online_update: OnlineUpdateSpec


def q_values_from_memristors(phi_s, synapses, ap_index):
    """
    Q(s,a) = sum_i phi[i] * W_i,a(ap_index)
    synapses: shape (D, A)
    """
    D = len(synapses)
    A = len(synapses[0])
    q = np.zeros(A, dtype=np.float32)
    for a in range(A):
        acc = 0.0
        for i in range(D):
            v = float(phi_s[i])
            if v != 0.0:
                acc += v * float(synapses[i][a].weight(ap_index=ap_index))
        q[a] = acc
    return q


def two_task_learning_step(
    tasks,
    # per task: (phi_s, action, reward, phi_s_next, terminated)
    experiences,
    synapses,  # (D, A)
    gamma,
    spec,
):
    """
    Paper rule: each learning step updates ALL tasks before proceeding. :contentReference[oaicite:3]{index=3}

    Update direction for each weight uses sign(delta * phi[i]) for the chosen action.
    Then apply paper pulse rule via apply_online_update().
    """
    D = len(synapses)

    for task in tasks:
        phi_s, action, reward, phi_s_next, terminated = experiences[task.name]

        q = q_values_from_memristors(phi_s, synapses, ap_index=task.ap_index)
        q_next = q_values_from_memristors(phi_s_next, synapses, ap_index=task.ap_index)

        q_sa = float(q[action])
        y = float(reward) if terminated else float(reward + gamma * np.max(q_next))
        delta = float(y - q_sa)

        for i in range(D):
            signal = delta * float(phi_s[i])
            direction = 1 if signal > 0 else (-1 if signal < 0 else 0)
            if direction != 0:
                apply_online_update(
                    synapse=synapses[i][action],
                    direction=direction,
                    ap_index=task.ap_index,
                    spec=spec.online_update,
                )
