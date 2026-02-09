import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from devices.multiWeightSynapse import MultiWeightSynapse
from devices.crosspointState import CrosspointState
from devices.multiWeightSynapse import MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams


def one_hot_to_index(state):
    return int(np.argmax(state))


class MemristiveQFunction:
    def __init__(self, state_size, action_size, spec, params):
        assert state_size>0, "State size must be greater than 0"
        assert action_size>0, "Action size must be greater than 0"
        self.n_state = state_size
        self.n_action = action_size
        self.q_table: List[List[MultiWeightSynapse]] = []
        for s in range(self.n_state):
            row: List[MultiWeightSynapse] = []
            for a in range(self.n_action):
                synapse = MultiWeightSynapse(spec,params)
                row.append(synapse)
            self.q_table.append(row)

    def get_synapse(self, state, action) -> MultiWeightSynapse:
        state_idx = one_hot_to_index(state)
        return self.q_table[state_idx][action]

    def read_q_value(self, state, action, ap_index):
        current_synapse = self.get_synapse(state, action)
        current_q = current_synapse.weight(ap_index)
        return current_q

    def update_q_value(self, state, action, sign_gradient: List[int]):
        current_synapse = self.get_synapse(state, action)
        for i, sign in enumerate(sign_gradient):
            if sign > 0:
                current_synapse.increase_bias_crosspoint_index()
            elif sign < 0:
                current_synapse.increase_positive_crosspoint_index(i)


