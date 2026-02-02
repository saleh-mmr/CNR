from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from envs.frozen_lake_env import FrozenLakeEnv, FrozenLakeSpec
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.online_update import OnlineUpdateSpec
from learning.multitask_step import (
    TaskSpec,
    MultiTaskUpdateSpec,
    multitask_learning_step,
)

# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainSpec:
    episodes: int = 3000
    max_steps_per_episode: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200

    gamma: float = 0.99
    seed: int = 0
    log_every: int = 100


def linear_epsilon(ep, spec):
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def moving_average(x, window=100):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ============================================================
# Memristor Q-network
# ============================================================

def build_memristor_q_network(
    n_states,
    n_actions,
    scaling_factor,
    mr_params,
    rng,
):

    syn_spec = MultiWeightSynapseSpec(
        n_plus=1,
        scaling_factor=scaling_factor,
    )

    synapses = []
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


def q_values_from_memristors(phi_s, synapses, ap_index):
    s_idx = int(np.argmax(phi_s))
    q = np.zeros(len(synapses[s_idx]), dtype=np.float32)
    for a in range(len(q)):
        q[a] = synapses[s_idx][a].weight(ap_index=ap_index)
    return q


def choose_action_eps_greedy(q, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q)))
    return int(np.argmax(q))


# ============================================================
# Main
# ============================================================

def main():
    train_spec = TrainSpec()
    rng = np.random.default_rng(train_spec.seed)

    env = FrozenLakeEnv(
        FrozenLakeSpec(
            map_name="4x4",
            is_slippery=True,
            seed=train_spec.seed,
        )
    )

    mr_params = MagnetoresistanceParams(
        a=1.0,
        b=0.0,
        c=0.1,
        g_threshold=0.5,
        sigma_pulse_noise=0.0,
        min_pulse_index_for_log=1,
    )

    synapses = build_memristor_q_network(
        env.n_states,
        env.n_actions,
        scaling_factor=1.0,
        mr_params=mr_params,
        rng=rng,
    )

    tasks = [TaskSpec(name="FL", ap_index=0)]
    update_spec = MultiTaskUpdateSpec(
        online_update=OnlineUpdateSpec(pulses_per_update=1)
    )

    # ========================================================
    # LOGGING (per episode)
    # ========================================================

    ep_x_plus = []
    ep_x_bias = []
    ep_W = []
    ep_reward = []

    # ========================================================
    # Training Loop
    # ========================================================

    for ep in range(train_spec.episodes):
        epsilon = linear_epsilon(ep, train_spec)
        s, phi_s, _ = env.reset()

        total_reward = 0.0
        x_plus_vals = []
        x_bias_vals = []
        W_vals = []

        for step in range(train_spec.max_steps_per_episode):
            q = q_values_from_memristors(phi_s, synapses, ap_index=0)
            a = choose_action_eps_greedy(q, epsilon, rng)

            s2, phi_s2, r, terminated, truncated, _ = env.step(a)
            total_reward += r

            experiences = {
                "FL": (phi_s, a, r, phi_s2, terminated)
            }

            multitask_learning_step(
                tasks=tasks,
                experiences=experiences,
                synapses=synapses,
                gamma=train_spec.gamma,
                spec=update_spec,
            )

            syn = synapses[int(np.argmax(phi_s))][a]
            x_plus_vals.append(syn.plus_states[0].x)
            x_bias_vals.append(syn.bias_state.x)
            W_vals.append(syn.weight(ap_index=0))

            phi_s = phi_s2
            if terminated or truncated:
                break

        ep_x_plus.append(np.mean(x_plus_vals))
        ep_x_bias.append(np.mean(x_bias_vals))
        ep_W.append(np.mean(W_vals))
        ep_reward.append(total_reward)

        if (ep + 1) % train_spec.log_every == 0:
            print(
                f"Episode {ep+1:5d} | "
                f"eps={epsilon:.3f} | "
                f"reward={total_reward:.2f}"
            )

    env.close()

    # ========================================================
    # Plotting
    # ========================================================

    episodes = np.arange(len(ep_reward))
    avg_reward_100 = moving_average(ep_reward, window=100)

    plt.figure(figsize=(12, 11))

    plt.subplot(4, 1, 1)
    plt.plot(episodes, ep_x_plus)
    plt.ylabel("x_plus")

    plt.subplot(4, 1, 2)
    plt.plot(episodes, ep_x_bias)
    plt.ylabel("x_bias")

    plt.subplot(4, 1, 3)
    plt.plot(episodes, ep_W)
    plt.ylabel("W")

    plt.subplot(4, 1, 4)
    plt.plot(episodes, ep_reward, alpha=0.4, label="Reward")
    if len(avg_reward_100) > 0:
        plt.plot(
            np.arange(99, 99 + len(avg_reward_100)),
            avg_reward_100,
            linewidth=2.5,
            label="Avg Reward (100)"
        )
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
