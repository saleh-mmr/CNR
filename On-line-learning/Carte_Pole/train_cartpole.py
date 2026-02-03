from __future__ import annotations
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dataclasses import dataclass
import numpy as np
from envs.carte_pole_env import CartPoleEnv
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.multitask_step import multitask_learning_step_cartpole

@dataclass(frozen=True)
class TaskSpec:
    name: str
    ap_index: int  # which '+' crosspoint is AP for this task


# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainSpec:
    episodes: int = 7000
    max_steps: int = 300

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 10000

    gamma: float = 0.99
    seed: int = 0
    log_every: int = 50


# ============================================================
# Utilities
# ============================================================

def linear_epsilon(ep: int, spec: TrainSpec) -> float:
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def choose_action_eps_greedy(q: np.ndarray, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


# ============================================================
# Memristor Q-network
# ============================================================

def build_q_network(
    n_features: int,
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
        for _ in range(n_features)
    ]


# ============================================================
# Training loop
# ============================================================

def train():
    spec = TrainSpec()
    rng = np.random.default_rng(spec.seed)

    env = CartPoleEnv(seed=spec.seed)

    mr_params = MagnetoresistanceParams(
        a=1,
        b=1,
        c=0.1,
        g_threshold=0.5,
        sigma_pulse_noise=0,
        min_pulse_index_for_log=1,
    )

    synapses = build_q_network(
        n_features=env.obs_dim * 2,
        n_actions=env.n_actions,
        scaling_factor=70,
        mr_params=mr_params,
        rng=rng,
    )

    tasks = [TaskSpec(name="CP", ap_index=0)]
    all_episode_rewards = []
    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------
    for ep in range(spec.episodes):
        epsilon = linear_epsilon(ep, spec)
        _, phi_s, _ = env.reset()

        episode_reward = 0.0

        for _ in range(spec.max_steps):
            # -------- act --------
            q = np.zeros(env.n_actions, dtype=np.float32)
            for i, phi_i in enumerate(phi_s):
                if phi_i != 0.0:
                    for a in range(env.n_actions):
                        w, _ = synapses[i][a].weight(ap_index=0)
                        q[a] += phi_i * float(w)

            action = choose_action_eps_greedy(q, epsilon, rng)

            # -------- step --------
            _, phi_s2, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            multitask_learning_step_cartpole(
                tasks=tasks,
                experiences={
                    "CP": (phi_s, action, reward, phi_s2, terminated)
                },
                synapses=synapses,
                gamma=spec.gamma,
            )

            phi_s = phi_s2
            if terminated or truncated:
                break
        all_episode_rewards.append(episode_reward)

        if (ep + 1) % spec.log_every == 0:
            print(
                f"Episode {ep+1:5d} | "
                f"eps={epsilon:.3f} | "
                f"reward={episode_reward:.1f}"
            )

    env.close()
    return all_episode_rewards


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    rewards = train()

    # Plot rewards and rolling mean (mean of last `window` episodes).
    window = 50
    rewards_arr = np.asarray(rewards, dtype=float)

    if rewards_arr.size == 0:
        print("No rewards to plot.")
    else:
        # Compute rolling mean where for the first (window-1) entries we use
        # the expanding mean (mean of available episodes), and afterwards
        # use the fixed-size window mean.
        cumsum = np.cumsum(rewards_arr)
        rolling_mean = np.empty_like(rewards_arr)
        for i in range(rewards_arr.size):
            start = max(0, i - window + 1)
            total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
            count = i - start + 1
            rolling_mean[i] = total / float(count)

        plt.figure()
        plt.plot(rewards_arr, label="Episode reward", alpha=0.4)
        plt.plot(rolling_mean, label=f"{window}-episode mean", color="C1")
        plt.title("CartPole Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        filename = "cartpole_training_rewards.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        # In headless environments plt.show() may block or be a no-op; keep it
        # but catch exceptions.
        try:
            plt.show()
        except Exception:
            pass
