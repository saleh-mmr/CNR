import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def apply_online_update(synapse, sign_gradient):
    for i, sign in enumerate(sign_gradient):
        if sign>0:
            synapse.increase_positive_crosspoint_index(i)
        elif sign<0:
            synapse.increase_bias_crosspoint_index()
