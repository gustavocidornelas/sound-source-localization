"""
Multi-Channel Spike Detection
=============================

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from scipy.linalg import block_diag
from scipy.signal import find_peaks

from lmlib.utils.generator import (
    gen_convolve,
    gen_sinusoidal,
    gen_exponential,
    gen_unit_impulse,
    gen_rand_walk,
    gen_baseline_sin,
    gen_multichannel,
)

K = 3000
M = 3
len_sp = 20
pos_sp = [500, 700, 900, 1200, 1700, 2010]
amp_sp_ch = [0.5, 1, 0.5, 0.8, 0.1]
decay_sp = 0.88
len_bl = 4 * len_sp


# signal generation
k = np.arange(K)
y_sp = gen_convolve(
    gen_unit_impulse(K, pos_sp),
    gen_sinusoidal(len_sp, len_sp) * gen_exponential(len_sp, decay_sp),
)
y = [
    y_sp * amp_sp_ch[num]
    + 0.1
    * (gen_rand_walk(K, seed=1000 - num) + gen_baseline_sin(K, k_period=int(K * 0.25)))
    for num in range(M)
]
y = gen_multichannel(y, "multi-observation")

# Create models
alssm_bl = lm.AlssmPoly(Q=3, C=[[1, 0, 0, 0]], label="baseline")
alssm_sp = lm.AlssmSin(2 * np.pi / len_sp, decay_sp, label="spike")

# Create Segments
g_bl = 500
g_sp = 5000
seg_left = lm.Segment(a=-len_bl, b=-1, g=g_bl, direction=lm.FORWARD, delta=-1)
seg_mid = lm.Segment(a=0, b=len_sp, g=g_sp, direction=lm.BACKWARD)
seg_right = lm.Segment(
    a=len_sp + 1, b=len_sp + 1 + len_bl, g=g_bl, direction=lm.BACKWARD, delta=len_sp
)

# Mapping
F = [[0, 1, 0], [1, 1, 1]]

# Costs
c_costs = lm.CCost((alssm_sp, alssm_bl), (seg_left, seg_mid, seg_right), F)

se_param = lm.SEParam()
se_param.filter(c_costs, y)
se_param.minimize()

H = block_diag([[1], [0]], np.eye(alssm_bl.N))
x = se_param.minimize_lin(H)

J = se_param.eval_se(x, ref=(k, K))
x_bl = np.copy(x)
x_bl[:, 0:2, :] = 0
J_bl = se_param.eval_se(x_bl, ref=(k, K))
J_sum = np.sum(J, axis=-1)
J_bl_sum = np.sum(J_bl, axis=-1)


lcr = -0.5 * np.log(J_sum / J_bl_sum)

peaks, _ = find_peaks(lcr, height=0.051, distance=30)

# plot
fig, axs = plt.subplots(
    5, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [1, 3, 1, 1, 1]}, sharex="all"
)

# window
wins = c_costs.window(ps=[1, 1, 1], ref=(peaks, K), merge_seg="sum")
axs[0].plot(k, wins, lw=0.5)

# signals
axs[1].plot(k, y[:, 0, :], c="grey", lw=0.5, label="$w_k$")

# lcr
axs[2].plot(k, lcr, c="red", lw=0.7, label="LCR")
axs[2].scatter(peaks, lcr[peaks])

# costs model
axs[3].plot(k, J, c="blue", lw=0.7, label="$J$")
axs[3].plot(k, J_sum, c="k", lw=0.5, label="$J_{sum}$")

# cost baseline
axs[4].plot(k, J_bl, c="green", lw=0.7, label="$J^{bl}$")
axs[4].plot(k, J_bl_sum, c="k", lw=0.5, label="$J^{bl}_{sum}$")

plt.legend()
plt.show()
