import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from devices.multiWeightSynapse import MultiWeightSynapse
from learning.oneStepUpdate import apply_online_update
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
        apply_online_update(current_synapse, sign_gradient)



# Example usage:
params = CrossPointParams(a=20, b=2.0, c=20, g_threshold=0.8, sigma_pulse_noise=20)
spec = MultiWeightSynapseSpec(n_problem=2, scaling_factor=1)
q_function = MemristiveQFunction(state_size=2, action_size=2, spec=spec, params=params)
print("Memristive Q-Function initialized.")
for s in range(2):
    for a in range(2):
        synapse = q_function.q_table[s][a]
        print(f"Synapse at state {s}, action {a} initialized with weights: {[synapse.weight(i) for i in range(2)]}")# Example usage:
state = [0, 1]  # One-hot encoded state
action = 1
ap_index = 1
current_q_value = q_function.read_q_value(state, action, ap_index)
print(f"Current Q-value: {current_q_value}")
sign_gradient = [-1,1]  # Example sign gradient for the update
q_function.update_q_value(state, action, sign_gradient)
sign_gradient = [-1,1]  # Example sign gradient for the update
q_function.update_q_value(state, action, sign_gradient)
sign_gradient = [-1,1]  # Example sign gradient for the update
ap_index = 1
q_function.update_q_value(state, action, sign_gradient)
sign_gradient = [-1,1]  # Example sign gradient for the update
q_function.update_q_value(state, action, sign_gradient)
sign_gradient = [-1,1]  # Example sign gradient for the update
q_function.update_q_value(state, action, sign_gradient)
updated_q_value = q_function.read_q_value(state, action, ap_index)
print(f"Updated Q-value: {updated_q_value}")
for s in range(2):
    for a in range(2):
        synapse = q_function.q_table[s][a]
        print(f"Synapse at state {s}, action {a} initialized with weights: {[synapse.weight(i) for i in range(2)]}")# Example usage:
