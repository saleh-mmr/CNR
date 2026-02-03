from __future__ import annotations
import numpy as np
from learning.online_update import apply_online_update


def multitask_learning_step_cartpole(tasks, experiences, synapses, gamma):
    """
    Performs ONE paper-style learning step across ALL tasks (CartPole version).

    Differences vs FrozenLake:
      - φ(s) is continuous
      - Q(s,a) = sum_i φ_i(s) * θ_{i,a}
      - ONLY ONE synapse is updated per step (hardware-faithful):
            choose feature with largest |φ_i|
    """
    for task in tasks:
        phi_s, action, reward, phi_s_next, terminated = experiences[task.name]

        # -------- Q(s,a) --------
        q_sa = 0.0
        for i, phi_i in enumerate(phi_s):
            if phi_i != 0.0:
                w, _ = synapses[i][action].weight(ap_index=task.ap_index)
                q_sa += phi_i * float(w)

        # -------- target --------
        if terminated:
            y = float(reward)
        else:
            q_next = np.zeros(len(synapses[0]), dtype=np.float32)
            for i, phi_i in enumerate(phi_s_next):
                if phi_i == 0.0:
                    continue
                for a in range(len(q_next)):
                    w, _ = synapses[i][a].weight(ap_index=task.ap_index)
                    q_next[a] += phi_i * float(w)

            y = float(reward + gamma * np.max(q_next))

        delta = float(y - q_sa)

        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

        # -------- choose ONE feature to update --------
        i_star = int(np.argmax(np.abs(phi_s)))

        apply_online_update(
            synapse=synapses[i_star][action],
            direction=direction,
            ap_index=task.ap_index,
        )
