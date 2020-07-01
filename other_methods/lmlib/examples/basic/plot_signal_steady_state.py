"""
Signal Approximation in Steady State
====================================

This example generates a polynomial autonomous linear state space model and a simple left sided segment. The
:class:`~lmlib.statespace.cost.CostSeg` calculates the parameters for the recursions and filters the signal.
In this exaple the filter uses the steady state to speed up approximation.
Boundary effects on the start and end of the signal are quite possible.
The examples plots the signal and its approximation.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import (
    gen_sinusoidal,
    gen_rand_walk,
    gen_wgn,
    gen_baseline_sin,
)

# Signal
K = 1000
k = np.arange(K)
y = gen_sinusoidal(K, k_period=400) + 0.1 * gen_rand_walk(K, seed=1000)

# Polynomial ALSSM
Q = 3
alssm_poly = lm.AlssmPoly(Q)

# Segment
segment = lm.Segment(a=-np.inf, b=0, direction=lm.FORWARD, g=20)

# CostSeg
costs = lm.CostSegment(alssm_poly, segment)

# filter signal and take the approximation
se_param = lm.SEParamSteadyState()
WSS = lm.get_steady_state_W(costs)
se_param.set_steady_state_W(WSS)
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

####################### COMPOSITE COST ########################


# Generating synthetic test signal
K = 1000
t = np.linspace(0, 4 * np.pi, K)
k = np.arange(K)
y = gen_sinusoidal(K, k_period=500) + gen_wgn(K, sigma=0.02, seed=1000)

# Setting up our model, here a polynomial ALSSM
Q = 2  # polynomial system order
alssm_poly_left = lm.AlssmPoly(
    Q, label="Poly"
)  # defining an ALSSM for a polynomial of label *Poly*
alssm_poly_right = lm.AlssmPoly(
    Q, label="Poly"
)  # defining an ALSSM for a polynomial of label *Poly*

# Setting up the model's segments, defining the window (incl. interval borders) applied to the model
g = 20  # window weight; corresponds to the number of samples included under the window
# left-side segment with exponentially increasing window over the  interval [-ininity, -1]
segment_left = lm.Segment(a=-200, b=-1, direction=lm.FORWARD, g=g)

# right-side segment with exponentially decreasing window over the interval [0, +ininity]
segment_right = lm.Segment(a=0, b=200, direction=lm.BACKWARD, g=g)

# CompositeCost
F = [
    [1, 0],
    [0, 1],
]  # mapping matrix assigning the models to the segments; here: our single model is mapped to both segments
costs = lm.CCost(
    (alssm_poly_left, alssm_poly_right), (segment_left, segment_right), F
)  # defining a composite cost for our model and windows.

# filter signal and take the approximation
se_param = lm.SEParamSteadyState(label="filter parameter for alssm poly 2 segments")
WSS = lm.get_steady_state_W(costs, F)
se_param.set_steady_state_W(WSS)
se_param.filter(costs, y)
xs = se_param.minimize()

y_hat = costs.eval(xs, f=[0, 1])

# displaying intermediate and final results
ks = [100, 400, 620]
wins = costs.window(
    ps=[1, 1], ref=(ks, K), merge_k="none", merge_seg="max-win"
)  # generate local window shapes
trajs = costs.trajectory(
    xs[ks], F, ref=(ks, K), merge_k="max-win", merge_seg="none", thd=0.05
)  # generate local model trajectories
print(trajs.shape)

# Plotting results
fig, axs = plt.subplots(3, 1, sharex="all", gridspec_kw={"height_ratios": [1, 2, 2]})

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)

axs[0].plot(k, wins, lw=1, ls="-", c="black", label="$w_L$ : window of seg. 0")

axs[0].legend(loc="upper right")
axs[0].set(title="Example: Windowed Signal Approximation and Filtering")
axs[0].set(ylabel="window")

axs[1].plot(k, y, lw=0.5, color="gray", label="$y$ : input signal")
axs[1].plot(k, trajs[:, 0], color="black", lw=1, linestyle="--")
axs[1].plot(k, trajs[:, 1], color="red", lw=1, linestyle="--")
axs[1].legend(loc="upper right")
axs[1].set(ylabel="local approx.")

axs[2].plot(k, y, lw=0.3, color="gray", label="$y$ : input signal")
axs[2].plot(k, y_hat, color="blue", lw=1, label=r"$\hat{y}$ : filtered signal")
axs[2].legend(loc="upper right")
axs[2].set(xlabel="time index $k$")
axs[2].set(ylabel="filtered")

plt.show()
