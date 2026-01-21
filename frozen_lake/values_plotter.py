import os
import numpy as np
import matplotlib.pyplot as plt

# data
a, b, sigma, base_scale = 1.566e-8, 0.350e-8, 1.7e-9, 9e7
x = np.arange(1, 200000)
values = (a * np.log10(x) + b).astype(np.float32)
values += np.random.normal(0, sigma, size=values.shape).astype(np.float32)
values *= base_scale

# greedy: keep the first point and any later point that exceeds the last kept by min_delta
def keep_increasing(x, values, min_delta=0.0):
    x = np.asarray(x)
    v = np.asarray(values)
    if v.size == 0:
        return x[:0], v[:0]
    kept_idx = [0]
    last = v[0]
    for i in range(1, v.size):
        if v[i] > last + min_delta:
            kept_idx.append(i)
            last = v[i]
    idx = np.array(kept_idx, dtype=int)
    return x[idx], v[idx]

# LIS (indices) — O(n log n)
def lis_indices(values, strictly=True):
    import bisect
    v = np.asarray(values)
    n = v.size
    if n == 0:
        return np.array([], dtype=int)
    parent = np.full(n, -1, dtype=int)
    tails, tails_vals = [], []
    for i, val in enumerate(v):
        pos = bisect.bisect_left(tails_vals, val) if strictly else bisect.bisect_right(tails_vals, val)
        if pos == len(tails):
            tails.append(i); tails_vals.append(val)
        else:
            tails[pos] = i; tails_vals[pos] = val
        parent[i] = tails[pos - 1] if pos > 0 else -1
    # reconstruct indices
    res = []
    k = tails[-1]
    while k != -1:
        res.append(k)
        k = parent[k]
    return np.array(res[::-1], dtype=int)

# plot
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 4))
plt.plot(x, values, '.', markersize=3, alpha=0.4, label='original')

kx, kv = keep_increasing(x, values, min_delta=0.0)
print(f"Original: {len(x)}, Greedy kept: {len(kx)}")
plt.plot(kx, kv, 'o', markersize=4, alpha=0.9, color='red', label='greedy')

lis_idx = lis_indices(values, strictly=True)
x_lis, v_lis = x[lis_idx], values[lis_idx]
print(f"LIS kept: {len(lis_idx)}")
plt.plot(x_lis, v_lis, 'o', markersize=4, alpha=0.9, color='green', label='LIS')

plt.xlabel('x')
plt.ylabel('value')
plt.title('Scaled values vs x (original, greedy, LIS)')
plt.grid(True)
plt.legend()
plt.tight_layout()
out_path = os.path.join('outputs', 'compare_base_scales_plot.png')
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
