# # Creating a test for the MemristiveQFunction class
#
# ___________________________imports___________________________
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.multiWeightSynapse import MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams
from learning.memristiveQFunction import MemristiveQFunction

current_state = [1, 0, 0]  # One-hot encoded state for state index 0
action = 0

# __________________________initialize params___________________________
# crosspoint parameters
a = 1.0
b = 0
c = 3
g_threshold = 0.08
sigma_pulse_noise = 0.0
params = CrossPointParams(a=a, b=b, c=c, g_threshold=g_threshold, sigma_pulse_noise=sigma_pulse_noise)

#  initialize MemristiveQFunction
n_problem = 2
scaling_factor = 10.0
state_size = 3
action_size = 2
spec = MultiWeightSynapseSpec(n_problem=n_problem, scaling_factor=scaling_factor)
q_function = MemristiveQFunction(state_size=state_size, action_size=action_size, spec=spec, params=params)
print("Memristive Q-Function initialized.")


'''
G_p = (a * log10(x) + b) + noise
G_ap = 1.0 + c * (G_p - G_threshold) ** (3.0 / 4.0))

(state,action) = (0,0)

iteration 1:
index_1 = 1 , index_2 = 1 , index_bias = 1
                        => G1_p = 0.0 , G1_ap = 0.0
                        => G2_p = 0.0 , G2_ap = 0.0
                        => G_bias = 0.0
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = 0.0
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = 0.0

iteration 2: (sign_1 +) (sign_2 0)
index_1 = 1 , index_2 = 1 , index_bias = 2
                        => G1_p = 0.0 , G1_ap = 0.0
                        => G2_p = 0.0 , G2_ap = 0.0
                        => G_bias = 0.301
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = -3.010
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = -3.010


iteration 3: (sign_1 +) (sign_2 +)
index_1 = 1 , index_2 = 1 , index_bias = 4
                        => G1_p = 0.0 , G1_ap = 0.0
                        => G2_p = 0.0 , G2_ap = 0.0
                        => G_bias = 0.6020
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = -6.020
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = -6.020


iteration 4: (sign_1 -) (sign_2 0)
index_1 = 2 , index_2 = 1 , index_bias = 4
                        => G1_p = 0.3010 , G1_ap = 0.5920
                        => G2_p = 0.0 , G2_ap = 0.0
                        => G_bias = 0.6020
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = -0.0991
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = -3.0102


iteration 5: (sign_1 -) (sign_2 -)
index_1 = 3 , index_2 = 2 , index_bias = 4
                        => G1_p = 0.47712 , G1_ap = 1.1931
                        => G2_p = 0.3010 , G2_ap = 0.5921
                        => G_bias = 0.6020
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = 8.9213
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = 4.6720


iteration 6: (sign_1 +) (sign_2 -)
index_1 = 3 , index_2 = 3 , index_bias = 5
                        => G1_p = 0.47712 , G1_ap = 1.1931
                        => G2_p = 0.4771 , G2_ap = 1.1931
                        => G_bias = 0.6989
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = 9.71320
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = 9.71320



iteration 7: (sign_1 +) (sign_2 -)
index_1 = 3 , index_2 = 4 , index_bias = 6
                        => G1_p = 0.47712 , G1_ap = 1.1931
                        => G2_p = 0.6020 , G2_ap = 1.71136
                        => G_bias = 0.77815
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = 10.1707
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = 14.1033


iteration 8: (sign_1 +) (sign_2 -)
index_1 = 3 , index_2 = 5 , index_bias = 7
                        => G1_p = 0.47712 , G1_ap = 1.1931
                        => G2_p = 0.6989 , G2_ap = 2.1622
                        => G_bias = 0.84509
                                    => weight_1 = 10.0 * (G1_ap + G2_p - G_bias) = 10.4704
                                    => weight_2 = 10.0 * (G1_p + G2_ap - G_bias) = 17.9428
'''







def log_iteration(q_function, current_state, action, sign_gradient, iteration):
    syn = q_function.get_synapse(current_state, action)

    # Update
    q_function.update_q_value(current_state, action, sign_gradient)

    # States
    idx_1 = syn.get_positive_crosspoint_state(0)
    idx_2 = syn.get_positive_crosspoint_state(1)
    idx_bias = syn.get_bias_crosspoint_state()

    # Conductances
    G1_p  = syn.get_positive_crosspoint_conductance_p(0)
    G1_ap = syn.get_positive_crosspoint_conductance_ap(0)
    G2_p  = syn.get_positive_crosspoint_conductance_p(1)
    G2_ap = syn.get_positive_crosspoint_conductance_ap(1)
    G_bias = syn.get_bias_crosspoint_conductance()

    # Weights
    w0 = syn.weight(0)
    w1 = syn.weight(1)

    # Clean structured print
    print(f"\n{'='*70}")
    print(f"Iteration {iteration} | sign_gradient = {sign_gradient}")
    print(f"{'-'*70}")
    print(f"Indices   : index_1={idx_1}, index_2={idx_2}, index_bias={idx_bias}")
    print(f"Gradients : "
          f"G1_p={G1_p:.4f}, G1_ap={G1_ap:.4f} | "
          f"G2_p={G2_p:.4f}, G2_ap={G2_ap:.4f} | "
          f"G_bias={G_bias:.4f}")
    print(f"Weights   : w0={w0:.4f}, w1={w1:.4f}")


sign_gradients = [
    [1, 0],    # iteration 2
    [1, 1],    # iteration 3
    [-1, 0],   # iteration 4
    [-1, -1],  # iteration 5
    [1, -1],   # iteration 6
    [1, -1],   # iteration 7
    [1, -1],   # iteration 8
    [1, -1],  # iteration 9
    [1, -1],  # iteration 10
    [1, 1],  # iteration 11
    [1, 1],  # iteration 12
    [-1, 1],  # iteration 13
    [1, 1],  # iteration 14
    [1, 1],  # iteration 15
    [1, 1],  # iteration 16
    [1, 1],  # iteration 17
    [1, 1],  # iteration 18
    [1, 1],  # iteration 19
    [1, 1],  # iteration 20
    [1, 1],  # iteration 21
    [1, 1],  # iteration 22
    [1, 1],  # iteration 23
    [1, 1],  # iteration 24
    [1, 1],  # iteration 25
    [1, 1],  # iteration 26
    [1, 1],  # iteration 27
    [1, -1],  # iteration 28
    [1, 1],  # iteration 29
    [1, 1],  # iteration 30
    [1, 1],  # iteration 31
    [1, 1],  # iteration 32
    [1, 1],  # iteration 33
    [-1, 1],  # iteration 34
    [1, 1],  # iteration 35
    [1, 1],  # iteration 36
    [1, 1],  # iteration 37
    [1, 1],  # iteration 38
    [1, 1],  # iteration 39
    [1, 1],  # iteration 40
    [1, 1],  # iteration 41
    [1, 1],  # iteration 42
    [1, 1],  # iteration 43
    [1, 1],  # iteration 44
    [1, 1],  # iteration 45
    [1, 1],  # iteration 46
    [1, 1],  # iteration 47
    [1, 1],  # iteration 48
    [1, 1],  # iteration 49
    [1, 1],  # iteration 50
    [1, 1],  # iteration 51
    [1, 1],  # iteration 52
    [1, 1],  # iteration 53
    [1, 1],  # iteration 54
    [1, 1],  # iteration 55
    [1, 1],  # iteration 56
    [1, 1],  # iteration 57
    [1, 1],  # iteration 58
    [1, 1],  # iteration 59
    [1, 1],  # iteration 60
    [1, 1],  # iteration 61
    [1, 1],  # iteration 62
    [1, -1],  # iteration 63
    [1, 1],  # iteration 64
    [1, 1],  # iteration 65
    [1, 1],  # iteration 66
    [1, 1],  # iteration 67
    [1, 1],  # iteration 68
    [1, 1],  # iteration 69
    [1, 1],  # iteration 70
    [1, 1],  # iteration 71
    [1, 1],  # iteration 72
    [1, 1],  # iteration 73
    [1, 1],  # iteration 74
    [1, 1],  # iteration 75
    [1, 1],  # iteration 76
    [1, 1],  # iteration 77
    [1, 1],  # iteration 78
    [1, 1],  # iteration 79
    [1, 1],  # iteration 80
    [1, 1],  # iteration 81
    [1, 1],  # iteration 82
    [1, 1],  # iteration 83
    [1, 1],  # iteration 84
    [1, 1],  # iteration 85
    [1, 1],  # iteration 86
    [1, 1],  # iteration 87
    [1, 1],  # iteration 88
    [1, 1],  # iteration 89
    [1, 1],  # iteration 90
    [-1, 1],  # iteration 91
    [1, 1],  # iteration 92
    [1, 1],  # iteration 93
    [1, 1],  # iteration 94
    [1, 1],  # iteration 95
    [1, 1],  # iteration 96
    [1, 1],  # iteration 97
    [1, 1],  # iteration 98
    [1, 1],  # iteration 99
    [1, 1],  # iteration 100





]

for i, sign in enumerate(sign_gradients, start=2):
    log_iteration(
        q_function=q_function,
        current_state=current_state,
        action=action,
        sign_gradient=sign,
        iteration=i
    )
