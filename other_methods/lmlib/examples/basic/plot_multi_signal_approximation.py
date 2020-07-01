"""
Multi Signal Approximation
==========================

Approximates a polynomial model to multiple signals simultaneously.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_rand_walk

# Signal
K = 1000
k = np.arange(K)
periods = [130, 150, 200, 220]
M = len(periods)
y = np.stack(
    [
        gen_sinusoidal(K, k_period=p) + 0.1 * gen_rand_walk(K, seed=1000)
        for p in periods
    ],
    axis=-1,
)[:, np.newaxis, :]

# Polynomial ALSSM
Q = 3
C = [[1, 0, 0, 0]]
alssm_poly = lm.AlssmPoly(Q, C)

# Segment
segment_left = lm.Segment(a=-np.inf, b=0, direction=lm.FORWARD, g=20)
segment_right = lm.Segment(a=1, b=np.inf, direction=lm.BACKWARD, g=20)

# CostSeg
ccost = lm.CCost((alssm_poly,), (segment_left, segment_right), F=[[1, 1]])

# filter signal and take the approximation
se_param = lm.SEParam()
se_param.filter(ccost, y)
xs = se_param.minimize()

y_hat = ccost.eval(xs, f=[1], ref=(k, K))

# Plot
fig, axs = plt.subplots(M, 1, sharex="all")
for m, ax in enumerate(axs):
    ax.plot(k, y[:, :, m], lw=0.6, c="gray", label=r"$y_{}$".format(m))
    ax.plot(k, y_hat[:, :, m], lw=1, label=r"$\hat{{y}}_{}$".format(m))
    ax.legend(loc="upper right")

axs[-1].set_xlabel("time")

plt.show()
