from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pickle

from envs.discrete_two_task_envs import (
    FrozenLakeAdapter, FrozenLakeAdapterSpec,
    CliffWalkingAdapter, CliffWalkingAdapterSpec
)
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.online_update import OnlineUpdateSpec
from learning.two_task_step import TaskSpec, TwoTaskUpdateSpec, two_task_learning_step, q_values_from_memristors
from rl.reward_scaling import RewardScaleSpec, scale_and_clip
from logging_helper.two_task_logger import TwoTaskLogger
from logging_helper.episode_reward_logger import EpisodeRewardLogger


@dataclass
class TrainSpec:
    global_steps: int = 100000  # one step = one FL transition + one CL transition, then update both
    gamma: float = 0.99
    seed: int = 0

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000

    log_every: int = 2000


def linear_epsilon(step, spec):
    if step >= spec.epsilon_decay_steps:
        return spec.epsilon_end
    t = step / max(1, spec.epsilon_decay_steps)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def choose_action_eps_greedy(q, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, len(q)))
    return int(np.argmax(q))


def build_synapses(D, A, N, scaling, mr, rng):
    spec = MultiWeightSynapseSpec(n_plus=N, scaling_factor=scaling)
    syn: List[List[MultiWeightSynapse]] = []
    for i in range(D):
        row = []
        for a in range(A):
            row.append(MultiWeightSynapse(spec=spec, params=mr, rng=rng))
        syn.append(row)
    return syn


def main():
    ts = TrainSpec()
    rng = np.random.default_rng(ts.seed)

    # Shared representation
    D = 48
    A = 4

    # Envs
    fl = FrozenLakeAdapter(FrozenLakeAdapterSpec(map_name="4x4", is_slippery=False, seed=ts.seed, feature_dim=D))
    cl = CliffWalkingAdapter(CliffWalkingAdapterSpec(seed=ts.seed, feature_dim=D))
    reward_specs = {
        "FL": RewardScaleSpec(scale=1.0, clip=1.0),  # keep FrozenLake as-is
        "CL": RewardScaleSpec(scale=0.01, clip=1.0),  # scale down CliffWalking
    }

    # Paper mapping: N=2 tasks stored in 2 '+' crosspoints
    tasks = [
        TaskSpec(name="FL", ap_index=0),
        TaskSpec(name="CL", ap_index=1),
    ]

    # Device params (replace with experimental values later)
    mr = MagnetoresistanceParams(
        a=1.0, b=0.0, c=0.1, g_threshold=0.5,
        sigma_pulse_noise=0.0,
        min_pulse_index_for_log=1,
    )
    scaling = 1.0

    synapses = build_synapses(D=D, A=A, N=2, scaling=scaling, mr=mr, rng=rng)
    update_spec = TwoTaskUpdateSpec(online_update=OnlineUpdateSpec(pulses_per_update=1))

    # Init states
    phi_fl, _ = fl.reset()
    phi_cl, _ = cl.reset()

    episodes_done = {"FL": 0, "CL": 0}
    returns = {"FL": 0.0, "CL": 0.0}
    reward_loggers = {
        "FL": EpisodeRewardLogger(),
        "CL": EpisodeRewardLogger(),
    }
    logger = TwoTaskLogger()

    for gstep in range(ts.global_steps):
        eps = linear_epsilon(gstep, ts)

        experiences: Dict[str, Tuple[np.ndarray, int, float, np.ndarray, bool]] = {}

        # ---- FrozenLake transition ----
        q_fl = q_values_from_memristors(phi_fl, synapses, ap_index=0)
        a_fl = choose_action_eps_greedy(q_fl, eps, rng)
        phi_fl2, r_fl_raw, term_fl, trunc_fl, _ = fl.step(a_fl)
        r_fl = scale_and_clip(r_fl_raw, reward_specs["FL"])
        returns["FL"] += r_fl
        done_fl = term_fl or trunc_fl
        experiences["FL"] = (phi_fl, a_fl, r_fl, phi_fl2, term_fl)
        if done_fl:
            reward_loggers["FL"].log(
                episode_index=episodes_done["FL"],
                reward=returns["FL"],
            )
            episodes_done["FL"] += 1
            phi_fl, _ = fl.reset()
            returns["FL"] = 0.0

        else:
            phi_fl = phi_fl2

        # ---- CliffWalking transition ----
        q_cl = q_values_from_memristors(phi_cl, synapses, ap_index=1)
        a_cl = choose_action_eps_greedy(q_cl, eps, rng)
        phi_cl2, r_cl_raw, term_cl, trunc_cl, _ = cl.step(a_cl)
        r_cl = scale_and_clip(r_cl_raw, reward_specs["CL"])
        returns["CL"] += r_cl
        done_cl = term_cl or trunc_cl
        experiences["CL"] = (phi_cl, a_cl, r_cl, phi_cl2, term_cl)
        if done_cl:
            reward_loggers["CL"].log(
                episode_index=episodes_done["CL"],
                reward=returns["CL"],
            )
            episodes_done["CL"] += 1
            phi_cl, _ = cl.reset()
            returns["CL"] = 0.0

        else:
            phi_cl = phi_cl2

        # ---- Paper: update BOTH tasks, then advance global step ---- :contentReference[oaicite:4]{index=4}
        two_task_learning_step(
            tasks=tasks,
            experiences=experiences,
            synapses=synapses,
            gamma=ts.gamma,
            spec=update_spec,
        )

        if (gstep + 1) % ts.log_every == 0:
            w_fl = synapses[0][0].weight(ap_index=0)
            w_cl = synapses[0][0].weight(ap_index=1)

            logger.log(
                global_step=gstep + 1,
                w_fl=w_fl,
                w_cl=w_cl,
                episodes_fl=episodes_done["FL"],
                episodes_cl=episodes_done["CL"],
            )

            print(
                f"GlobalStep {gstep + 1:7d} | eps={eps:.3f} | "
                f"episodes FL/CL={episodes_done['FL']}/{episodes_done['CL']} | "
                f"W00 FL/CL={w_fl:.3f}/{w_cl:.3f}"
            )

    fl.close()
    cl.close()
    with open("analysis/two_task_log.pkl", "wb") as f:
        pickle.dump(logger, f)

    with open("analysis/episode_rewards_fl.pkl", "wb") as f:
        pickle.dump(reward_loggers["FL"], f)

    with open("analysis/episode_rewards_cl.pkl", "wb") as f:
        pickle.dump(reward_loggers["CL"], f)

if __name__ == "__main__":
    main()

