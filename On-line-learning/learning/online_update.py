from __future__ import annotations

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from devices.multiweight_synapse import MultiWeightSynapse


@dataclass
class OnlineUpdateSpec:
    """
    On-line learning control parameters.
    """
    pulses_per_update: int = 1   # usually 1, but kept explicit


def apply_online_update(
    synapse: MultiWeightSynapse,
    direction: int,
    ap_index: int,
    spec: OnlineUpdateSpec,
) -> None:
    """
    Apply ONE on-line update to ONE composite synapse, exactly per the paper.

    Parameters
    ----------
    synapse : MultiWeightSynapse
        The composite memristive synapse (one ANN weight).
    direction : int
        +1 -> weight should increase
        -1 -> weight should decrease
         0 -> no change
    ap_index : int
        Which '+' crosspoint corresponds to the current task (AP state).
        For FrozenLake-only: ap_index = 0.
    spec : OnlineUpdateSpec

    Paper rules:
      - If weight needs to be increased:
            increase x_n (corresponding '+' crosspoint)
      - If weight needs to be decreased:
            increase x_B (bias crosspoint)
      - Redraw noise when x changes
    """
    if direction > 0:
        # Increase corresponding '+' crosspoint
        synapse.increase_plus(ap_index)

    elif direction < 0:
        # Decrease weight by increasing bias
        synapse.increase_bias()

    # direction == 0 -> do nothing


def batch_online_update(
    synapses: List[MultiWeightSynapse],
    directions: List[int],
    ap_index: int,
    spec: OnlineUpdateSpec,
) -> None:
    """
    Apply on-line updates to a batch of synapses (e.g., all weights for one action).

    synapses[i] gets directions[i].
    """
    assert len(synapses) == len(directions)

    for syn, d in zip(synapses, directions):
        apply_online_update(
            synapse=syn,
            direction=int(d),
            ap_index=ap_index,
            spec=spec,
        )
