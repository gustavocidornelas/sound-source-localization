r"""
.. _lmlib_composite_cost_example:

Composite Cost
==============

This example demonstrates the application of a composite cost :class:`CCost` to:

* locally approximate a single-channel discrete-time signal,
* filter a single-channel signal with a symmetric linear
  filter; the filter output is :math:`\hat{y}_k = c x_k` for any sample index :math:`k`.

In both applications :class:`CCost` is a single polynomial :class:`AlssmPoly` model of oder 4 and spanning over two
segments :class:`Segment`.

.. note:: For more details also see the tutorial examples.

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_wgn, gen_baseline_sin

# Generating synthetic test signal
K = 1000
t = np.linspace(0, 4 * np.pi, K)
k = np.arange(K)
y = gen_sinusoidal(K, k_period=500) + gen_wgn(K, sigma=0.02, seed=1000)

# Setting up our model, here a polynomial ALSSM
Q = 3  # polynomial system order
alssm_poly_left = lm.AlssmPoly(
    Q, label="Poly"
)  # defining an ALSSM for a polynomial of label *Poly*
alssm_poly_right = lm.AlssmPoly(
    Q, label="Poly"
)  # defining an ALSSM for a polynomial of label *Poly*


# Setting up the model's segments, defining the window (incl. interval borders) applied to the model
g = 35  # window weight; corresponds to the number of samples included under the window
# left-side segment with exponentially increasing window over the  interval [-ininity, -1]
segment_left = lm.Segment(a=-np.inf, b=-1, direction=lm.FORWARD, g=g)

# right-side segment with exponentially decreasing window over the interval [0, +ininity]
segment_right = lm.Segment(a=0, b=np.inf, direction=lm.BACKWARD, g=g)

# CompositeCost
F = [
    [1, 0],
    [0, 1],
]  # mapping matrix assigning the models to the segments; here: our single model is mapped to both segments
costs = lm.CCost(
    (alssm_poly_left, alssm_poly_right), (segment_left, segment_right), F
)  # defining a composite cost for our model and windows.

# filter signal and take the approximation
f_param = lm.SEParam(label="filter parameter for alssm poly 2 segments")
f_param.filter(costs, y)
xs = f_param.minimize()

y_hat = costs.eval(xs, f=[1, 0])


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
