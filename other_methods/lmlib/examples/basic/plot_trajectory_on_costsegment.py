"""
.. _lmlib_example_trajectory_on_costsegment:

Trajectory on CostSegment
=========================

This example defines a cost segment using a discrete-time autonomous linear state space model (ALSSM) and a left-sided,
exponentially decaying window. See: :class:`~lmlib.statespace.costfunc.Segment`,
:class:`~lmlib.statespace.costfunc.CostSegment`, :class:`~lmlib.statespace.model.Alssm`.

"""
import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_wgn, gen_baseline_sin

# generate signal
K = 500
k = np.arange(K)
y = (
    gen_sinusoidal(K, k_period=100)
    + gen_wgn(K, sigma=0.02, seed=1000)
    + gen_baseline_sin(K, k_period=1000)
)

# setup model, segment and cost segment
alssm_poly = lm.AlssmPoly(Q=2)
segment = lm.Segment(a=-np.inf, b=0, direction=lm.FORWARD, g=5)
cost_segment = lm.CostSegment(alssm_poly, segment)

# setup cost function and minimize
se_parma = lm.SEParam()
se_parma.filter(cost_segment, y)
xs = se_parma.minimize()

# -----------------  PLOT  -----------------
ks = [85, 100, 400]
trajs = cost_segment.trajectory(xs[ks], ref=(ks, K), merge_k="none", thd=0.0001)

# get merged window weights at ks
wins = cost_segment.window(ref=(ks, K), merge_k="max-win")

fig, axs = plt.subplots(2, 1, sharex="all")

axs[0].plot(k, wins, c="black", label="$w_i(k_s)$")
axs[0].set_ylabel("window weights")
axs[0].legend()

axs[1].plot(k, y, lw=0.5, c="grey", label="y")
axs[1].plot(k, trajs, lw=1.3, label="$s_i(k_s)$")
axs[1].set_xlabel("time")
axs[1].set_ylabel("signal")
axs[1].legend()

plt.show()
