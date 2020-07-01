"""
.. _lmlib_example_trajectory_on_compositecost:

Trajectory on CompositeCost
===========================

This example demonstrates the use of a composite cost composed of two segments, a left- and
right-sided exponentially decaying window, and a single ALSSM, a polynomial.
See: :class:`~lmlib.statespace.costfunc.Segment`, :class:`~lmlib.statespace.costfunc.CCost`, :class:`~lmlib.statespace.model.Alssm`.
"""

##############################
# Model Composition and Model Fit
# -------------------------------


import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_wgn, gen_baseline_sin

# Signal generation
K = 600
k = np.arange(K)
y = (
    gen_sinusoidal(K, k_period=140)
    + gen_wgn(K, sigma=0.1, seed=1000)
    + gen_baseline_sin(K, k_period=1000)
)

# Model composition
alssm_poly = lm.AlssmPoly(Q=3)
segment_left = lm.Segment(a=-np.inf, b=-1, direction=lm.FORWARD, g=10, delta=0)
segment_right = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=10, delta=0)
F = [[1, 1]]  # mapping matrix
c_cost = lm.CCost(
    (alssm_poly,), (segment_left, segment_right), F
)  # composite cost of two segments with one common model

# Cost function and cost minimization (=model fitting)
se_param = lm.SEParam()
se_param.filter(c_cost, y)
xs = se_param.minimize()


##############################
# Merged-Segments Plotting: Window Weights & Model Trajectories
# -------------------------------------------------------------
# Generation of multiple windowed trajectories, each spanning *over all segments* and cropped by a window threshold value:
# :code:`trajs = c_cost.trajectories_over(K, k0s, xs[k0s], F=F, merge_k='none', merge_seg='max-win', thd=0.0001)`

k0s = [100, 300, 400]

# get merged window weights at ks
wins = c_cost.window(ps=[1, 1], ref=(k0s, K), merge_k="max-win", merge_seg="max-win")

# get trajectories per segment at indices k0s
trajs = c_cost.trajectory(
    xs[k0s], F=F, ref=(k0s, K), merge_k="none", merge_seg="max-win", thd=0.1
)

fig, axs = plt.subplots(2, 1, sharex="all")
axs[0].set_title("merged windows over all segments at indices $k0$".format(k0s))
axs[0].plot(k, wins, lw=0.8, c="black")
axs[0].set_ylabel("window weights")
axs[0].legend()

axs[1].plot(k, y, lw=0.5, c="grey", label="y")
axs[1].set_title("merged trajectories over all segments at indices $k0$")
for ii, traj in enumerate(trajs.T):
    axs[1].plot(k, traj, lw=1.5, label="$k0 = {}$".format(k0s[ii]))
axs[1].set_xlabel("index $k$")
axs[1].set_ylabel("y")
axs[1].legend()

plt.show()
