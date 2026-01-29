from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from envs.frozen_lake_env import FrozenLakeEnv, FrozenLakeSpec
from rl.q_learning_target import QLearningSpec
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.online_update import OnlineUpdateSpec, apply_online_update


@dataclass
class TrainSpec:
    episodes: int = 3700
    max_steps_per_episode: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200  # linear decay
    gamma: float = 0.99
    seed: int = 0

    # Paper mapping (FrozenLake only => N=1)
    ap_index_for_frozenlake: int = 0

    # Logging
    log_every: int = 100


def linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    if episode >= decay_episodes:
        return float(end)
    t = episode / max(decay_episodes, 1)
    return float(start + t * (end - start))


def build_memristor_q_network(
    n_states: int,
    n_actions: int,
    scaling_factor: float,
    mr_params: MagnetoresistanceParams,
    rng: np.random.Generator,
) -> List[List[MultiWeightSynapse]]:
    """
    Create one composite synapse per parameter theta[s,a].

    For FrozenLake-only (one task), we set N=1 '+' crosspoint per synapse.
    Later you can set N=3 for CP/MC/FL exactly as the paper envisions. :contentReference[oaicite:1]{index=1}
    """
    syn_spec = MultiWeightSynapseSpec(n_plus=1, scaling_factor=scaling_factor)

    synapses = []
    for s in range(n_states):
        row = []
        for a in range(n_actions):
            row.append(MultiWeightSynapse(spec=syn_spec, params=mr_params, rng=rng))
        synapses.append(row)
    return synapses


def q_values_from_memristors(
    phi_s: np.ndarray,
    synapses: List[List[MultiWeightSynapse]],
    ap_index: int,
) -> np.ndarray:
    """
    Compute Q(s,a) = sum_i phi_i(s) * W_i,a
    For one-hot phi, this picks the weights for the active state.

    IMPORTANT: We do NOT keep a floating theta. We read weights from devices.
    """
    s_idx = int(np.argmax(phi_s))
    n_actions = len(synapses[s_idx])
    q = np.zeros(n_actions, dtype=np.float32)
    for a in range(n_actions):
        q[a] = float(synapses[s_idx][a].weight(ap_index=ap_index))
    return q


def choose_action_eps_greedy(q: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < float(epsilon):
        return int(rng.integers(0, len(q)))
    return int(np.argmax(q))


def main():
    # -------------------------
    # 1) Environment
    # -------------------------
    train_spec = TrainSpec()
    env = FrozenLakeEnv(FrozenLakeSpec(map_name="4x4", is_slippery=True, seed=train_spec.seed))
    rng = np.random.default_rng(train_spec.seed)

    # -------------------------
    # 2) Paper device parameters
    # -------------------------
    # NOTE: The paper says a,b,c,Gthreshold come from experiments; you must set real values later.
    # These are placeholder numbers so the code runs end-to-end.
    mr_params = MagnetoresistanceParams(
        a=1.0,
        b=0.0,
        c=0.1,
        g_threshold=0.5,
        sigma_pulse_noise=0.0,   # start deterministic; enable later
        min_pulse_index_for_log=1,
    )

    scaling_factor = 1.0  # start simple; tune later based on desired Q magnitude

    # -------------------------
    # 3) Build memristor Q "weights"
    # -------------------------
    synapses = build_memristor_q_network(
        n_states=env.n_states,
        n_actions=env.n_actions,
        scaling_factor=scaling_factor,
        mr_params=mr_params,
        rng=rng,
    )

    # Specs
    qspec = QLearningSpec(gamma=train_spec.gamma)
    uspec = OnlineUpdateSpec(pulses_per_update=1)

    # -------------------------
    # 4) Training loop (paper on-line update semantics)
    # -------------------------
    returns_window: List[float] = []

    for ep in range(train_spec.episodes):
        epsilon = linear_epsilon(
            ep,
            start=train_spec.epsilon_start,
            end=train_spec.epsilon_end,
            decay_episodes=train_spec.epsilon_decay_episodes,
        )

        s, phi_s, _ = env.reset()
        ep_return = 0.0

        for t in range(train_spec.max_steps_per_episode):
            # Read Q from memristors
            q = q_values_from_memristors(phi_s, synapses, ap_index=train_spec.ap_index_for_frozenlake)

            # Choose action
            a = choose_action_eps_greedy(q, epsilon, rng)

            # Step environment
            s2, phi_s2, r, terminated, truncated, _ = env.step(a)
            ep_return += r

            # Compute TD error using current memristor-based Q
            # We need q_sa and target. We'll compute target using q(s',·) from memristors.
            q_next = q_values_from_memristors(phi_s2, synapses, ap_index=train_spec.ap_index_for_frozenlake)
            q_sa = float(q[a])
            y = float(r) if terminated else float(r + train_spec.gamma * np.max(q_next))
            delta = float(y - q_sa)

            # Convert TD error sign into paper pulse updates:
            #   delta > 0 => increase selected weight
            #   delta < 0 => decrease selected weight (via bias)
            direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

            # Apply to the SINGLE parameter involved: theta[s,a]
            # With one-hot features, only the active state row updates.
            active_state = int(np.argmax(phi_s))
            apply_online_update(
                synapse=synapses[active_state][a],
                direction=direction,
                ap_index=train_spec.ap_index_for_frozenlake,
                spec=uspec,
            )

            # Move
            s, phi_s = s2, phi_s2

            if terminated or truncated:
                break

        returns_window.append(ep_return)
        if len(returns_window) > 200:
            returns_window.pop(0)

        if (ep + 1) % train_spec.log_every == 0:
            avg_ret = float(np.mean(returns_window))
            print(
                f"Episode {ep+1:5d} | eps={epsilon:.3f} | "
                f"avg_return(200)={avg_ret:.3f} | last_return={ep_return:.1f}"
            )

    env.close()


if __name__ == "__main__":
    main()
