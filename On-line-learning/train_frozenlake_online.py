from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

from envs.frozen_lake_env import FrozenLakeEnv, FrozenLakeSpec
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.online_update import OnlineUpdateSpec
from learning.multitask_step import (
    TaskSpec,
    MultiTaskUpdateSpec,
    multitask_learning_step,
)

# Import the local PulseLogger (not the stdlib `logging_helper` module)
from logging_helper.pulse_logger import PulseLogger

import pickle


# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainSpec:
    episodes: int = 2000
    max_steps_per_episode: int = 100

    # ε-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200

    gamma: float = 0.99
    seed: int = 0

    # Logging
    log_every: int = 100


def linear_epsilon(ep: int, spec: TrainSpec) -> float:
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


# ============================================================
# Memristor Q-network construction
# ============================================================

def build_memristor_q_network(
    n_states: int,
    n_actions: int,
    scaling_factor: float,
    mr_params: MagnetoresistanceParams,
    rng: np.random.Generator,
) -> List[List[MultiWeightSynapse]]:
    """
    One composite synapse per Q-table parameter θ[s,a].

    FrozenLake-only:
      N = 1 '+' crosspoint per synapse
    """
    syn_spec = MultiWeightSynapseSpec(
        n_plus=1,
        scaling_factor=scaling_factor,
    )

    synapses: List[List[MultiWeightSynapse]] = []
    for s in range(n_states):
        row = []
        for a in range(n_actions):
            row.append(
                MultiWeightSynapse(
                    spec=syn_spec,
                    params=mr_params,
                    rng=rng,
                )
            )
        synapses.append(row)

    return synapses


def q_values_from_memristors(
    phi_s: np.ndarray,
    synapses: List[List[MultiWeightSynapse]],
    ap_index: int,
) -> np.ndarray:
    """
    Q(s,a) readout from memristors.
    One-hot features => select active state row.
    """
    s_idx = int(np.argmax(phi_s))
    n_actions = len(synapses[s_idx])

    q = np.zeros(n_actions, dtype=np.float32)
    for a in range(n_actions):
        q[a] = synapses[s_idx][a].weight(ap_index=ap_index)

    return q


def choose_action_eps_greedy(
    q: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q)))
    return int(np.argmax(q))


# ============================================================
# Main training loop
# ============================================================

def main():
    # -----------------------------
    # 1) Setup
    # -----------------------------
    train_spec = TrainSpec()
    rng = np.random.default_rng(train_spec.seed)

    env = FrozenLakeEnv(
        FrozenLakeSpec(
            map_name="4x4",
            is_slippery=True,
            seed=train_spec.seed,
        )
    )

    # -----------------------------
    # 2) Device parameters (paper)
    # -----------------------------
    # NOTE: Replace with experimental values later
    mr_params = MagnetoresistanceParams(
        a=1.0,
        b=0.0,
        c=0.1,
        g_threshold=0.5,
        sigma_pulse_noise=0.0,     # start deterministic
        min_pulse_index_for_log=1,
    )

    scaling_factor = 1.0

    synapses = build_memristor_q_network(
        n_states=env.n_states,
        n_actions=env.n_actions,
        scaling_factor=scaling_factor,
        mr_params=mr_params,
        rng=rng,
    )

    # -----------------------------
    # 3) Paper multitask setup
    # -----------------------------
    tasks = [
        TaskSpec(name="FL", ap_index=0),  # FrozenLake stored in '+' crosspoint 0
    ]

    update_spec = MultiTaskUpdateSpec(
        online_update=OnlineUpdateSpec(pulses_per_update=1)
    )

    # -----------------------------
    # 4) Training
    # -----------------------------
    returns_window: List[float] = []

    for ep in range(train_spec.episodes):
        epsilon = linear_epsilon(ep, train_spec)

        s, phi_s, _ = env.reset()
        ep_return = 0.0

        for step in range(train_spec.max_steps_per_episode):
            # Read Q from memristors
            q = q_values_from_memristors(
                phi_s, synapses, ap_index=0
            )

            # Choose action
            a = choose_action_eps_greedy(q, epsilon, rng)

            # Environment step
            s2, phi_s2, r, terminated, truncated, _ = env.step(a)
            ep_return += r

            # Build experience dict (paper: all tasks per step)
            experiences: Dict[str, Tuple[np.ndarray, int, float, np.ndarray, bool]] = {
                "FL": (phi_s, a, r, phi_s2, terminated)
            }

            # === PAPER-CORRECT ONLINE LEARNING STEP ===
            multitask_learning_step(
                tasks=tasks,
                experiences=experiences,
                synapses=synapses,
                gamma=train_spec.gamma,
                spec=update_spec,
            )
            # ==========================================
            # ---- PHYSICS LOGGING (paper-critical) ----
            syn = synapses[int(np.argmax(phi_s))][a]

            pulse_logger.log(
                episode=ep,
                step=step,
                x_plus=syn.plus_states[0].x,  # FL crosspoint
                x_bias=syn.bias_state.x,
                weight=syn.weight(ap_index=0),
                reward=r,
            )

            phi_s = phi_s2
            if terminated or truncated:
                break

        returns_window.append(ep_return)
        if len(returns_window) > 200:
            returns_window.pop(0)

        if (ep + 1) % train_spec.log_every == 0:
            avg_ret = float(np.mean(returns_window))

            # Inspect ONE representative synapse (state 0, action 0)
            syn = synapses[0][0]

            print(
                f"Episode {ep + 1:5d} | "
                f"eps={epsilon:.3f} | "
                f"avg_return(200)={avg_ret:.3f} | "
                f"x_plus={syn.plus_states[0].x} | "
                f"x_bias={syn.bias_state.x} | "
                f"W={syn.weight(ap_index=0):.3f}"
            )

    env.close()


if __name__ == "__main__":
    pulse_logger = PulseLogger()
    main()
    with open("analysis/pulse_log.pkl", "wb") as f:
        pickle.dump(pulse_logger, f)
