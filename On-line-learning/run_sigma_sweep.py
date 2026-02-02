from __future__ import annotations

from dataclasses import dataclass
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
    max_steps: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200

    gamma: float = 0.99
    seed: int = 0
    log_every: int = 100


# ============================================================
# Utilities
# ============================================================

def linear_epsilon(ep: int, spec: TrainSpec) -> float:
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def moving_average(x, window: int):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


def choose_action_eps_greedy(q: np.ndarray, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


# ============================================================
# Memristor Q-network
# ============================================================

def build_q_network(
    n_states: int,
    n_actions: int,
    scaling_factor: float,
    mr_params: MagnetoresistanceParams,
    rng,
):
    syn_spec = MultiWeightSynapseSpec(
        n_plus=1,
        scaling_factor=scaling_factor,
    )

    return [
        [
            MultiWeightSynapse(syn_spec, mr_params, rng)
            for _ in range(n_actions)
        ]
        for _ in range(n_states)
    ]


def read_q(phi_s: np.ndarray, synapses, ap_index: int) -> np.ndarray:
    s_idx = int(np.argmax(phi_s))
    return np.array(
        [synapses[s_idx][a].weight(ap_index) for a in range(len(synapses[s_idx]))],
        dtype=np.float32,
    )


# ============================================================
# Training loop
# ============================================================

def train(sigma_pulse_noise: float):
    spec = TrainSpec()
    rng = np.random.default_rng(spec.seed)

    env = FrozenLakeEnv(
        FrozenLakeSpec(
            map_name="4x4",
            is_slippery=True,
            seed=spec.seed,
        )
    )

    mr_params = MagnetoresistanceParams(
        a=1.566e-8,
        b=0.350e-8,
        c=0.1,
        g_threshold=0.5,
        sigma_pulse_noise=sigma_pulse_noise,
        min_pulse_index_for_log=1,
    )

    synapses = build_q_network(
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

    rewards = []

    for ep in range(spec.episodes):
        epsilon = linear_epsilon(ep, spec)
        _, phi_s, _ = env.reset()
        total_reward = 0.0

        for _ in range(spec.max_steps):
            q = read_q(phi_s, synapses, ap_index=0)
            action = choose_action_eps_greedy(q, epsilon, rng)

            _, phi_s2, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            multitask_learning_step(
                tasks=tasks,
                experiences={
                    "FL": (phi_s, action, reward, phi_s2, terminated)
                },
                synapses=synapses,
                gamma=spec.gamma,
                spec=update_spec,
            )

            phi_s = phi_s2
            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()

    return moving_average(rewards, window=50)


def run_sigma_sweep():
    sigmas = [
        # 1.7e-14,
        1.7e-13,
        1.7e-12,
        1.7e-11,
        1.7e-10,
        1.7e-9,
        1.7e-8,
        # 1.7e-7,
        # 1.7e-6
    ]
    curves = {}

    for sigma in sigmas:
        print(f"Training with sigma = {sigma:.1e}")
        curves[sigma] = train(sigma)

    return curves


# ============================================================
# Plotting
# ============================================================



# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    curves = run_sigma_sweep()
    for sigma, values in curves.items():
        plt.plot(values, label=f"σ = {sigma:.1e}")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plot_sigma_comparison(curves)
