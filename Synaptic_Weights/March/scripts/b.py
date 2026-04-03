import numpy as np

goal_1 = +3.9
goal_2 = -1.5

scale = 7e9
tolerance = 1e-1
index_1 = 1.0
index_2 = 1.0
index_bias = 1.0

def weight(ap_index):
    if ap_index == 0:
        return scale * (ap(index_1) + p(index_2) - p(index_bias))
    else:
        return scale * (ap(index_2) + p(index_1) - p(index_bias))


def ap(index):
    a = 1.566e-8
    b = 3.5e-9
    g_s = 4.32e-7
    g_p = float(a * np.log10(index) + b)
    g_ap = g_p * (1.0 + (g_p / g_s) ** (3.0 / 4.0))
    return g_ap


def p(index):
    a = 1.566e-8
    b = 3.5e-9
    return  float(a * np.log10(index) + b)


counter = 0
while abs(weight(0) - goal_1) > tolerance or abs(weight(1) - goal_2) > tolerance:
    counter += 1
    print(f"Current weight for AP 0: {weight(0):.4f}, Current weight for AP 1: {weight(1):.8f}, index_1: {index_1:.4f}, index_2: {index_2:.4f}, index_bias: {index_bias:.4f}")
    if weight(0) < goal_1:
        index_1 += 1.0
    elif weight(0) > goal_1:
        index_bias += 1.0

    if weight(1) < goal_2:
        index_2 += 1.0
    elif weight(1) > goal_2:
        index_bias += 1.0

    current_weight_0 = weight(0)
    current_weight_1 = weight(1)

print(f"Final weight for AP 0: {weight(0):.4f}, Final weight for AP 1: {weight(1):.4f}, Total iterations: {counter}")