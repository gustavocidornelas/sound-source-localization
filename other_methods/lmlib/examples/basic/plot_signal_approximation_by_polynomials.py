"""
Signal Approximation by polynomial ALSSM
========================================

This example generates a polynomial autonomous linear state space model and a simple left sided segment. The
:class:`~lmlib.statespace.cost.CostSeg` calculates the parameters for the recursions and filters the signal.
The examples plots the signal and its approximation.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_rand_walk

# Signal
K = 1000
k = np.arange(K)
y = gen_sinusoidal(K, k_period=400) + 0.1 * gen_rand_walk(K, seed=1000)

# Polynomial ALSSM
Q = 3
alssm_poly = lm.AlssmPoly(Q)

# Segment
segment = lm.Segment(a=0, b=100, direction=lm.BACKWARD, g=80)

# CostSeg
costs = lm.CostSegment(alssm_poly, segment)

# filter signal and take the approximation
se_param = lm.SEParam()
se_param.filter(costs, y)
xs = se_param.minimize()


# ----------------  Plot  -----------------
ks = [150, 600]
wins = costs.window(ref=(ks, K))
trajs = costs.trajectory(xs[ks], ref=(ks, K), thd=0.1)

y_hat = costs.alssm.eval(xs[k], ref=(ks, K))

fig, axs = plt.subplots(2, 1, sharex="all")
axs[0].plot(k, wins, lw=1, c="k", ls="-")
axs[0].set_ylabel("window weights")

axs[1].plot(k, y, lw=0.5, c="grey")
axs[1].plot(k, y_hat, lw=0.7, c="k")
axs[1].plot(k, trajs, lw=1.5, c="r")
axs[1].set(xlabel="time", ylabel="voltage")
plt.show()
