from __future__ import annotations
import numpy as np


def read_q(phi_s, synapses, ap_index):
    """
    Read Q(s, a) from memristive synapses using linear function approximation:

        Q(s,a) = sum_i phi_i(s) * theta_{i,a}

    Parameters
    ----------
    phi_s : np.ndarray
        Feature vector φ(s), shape (n_features,)
    synapses : list[list[MultiWeightSynapse]]
        synapses[i][a] corresponds to feature i, action a
    ap_index : int
        Which '+' crosspoint is AP for the task
    """
    n_actions = len(synapses[0])
    q = np.zeros(n_actions, dtype=np.float32)

    for i, phi_i in enumerate(phi_s):
        if phi_i == 0.0:
            continue

        for a in range(n_actions):
            w, _ = synapses[i][a].weight(ap_index)
            q[a] += phi_i * float(w)

    return q
