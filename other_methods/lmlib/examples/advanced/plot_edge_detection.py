# -*- coding: utf-8 -*-
# Author: Wildhaber Reto, Waldmann Frédéric
r"""
Example: Edge Detection
=======================

This example demonstrates the detection of edges in a single-channel signal using autonomous linear state space models
:class:`~lmlib.statespace.model.Alssm`.
This example uses straight-line models to approximate the signals to the left and the right of the assumed edges.

To detect an edge at a specific time index :math:`k`, two :class:`~lmlib.statespace.model.Alssm` models are locally fit
to the given signal:
a first straight-model is fit to left of time index :math:`k` and a second straight-line model to the right of
:math:`k`.
The two models are fit under two different assumptions (also known as hypotheses):
First, the two lines are assumed to be continuous but to have individual slopes.
Second, the two lines are assumed to be continuous and of common slope.

Then, the edge probability at index :math:`k` is measured by the log-cost ratio *LCR*

.. math::
      LCR_k = -0.5  \ln \frac{J_k\big(\hat{x}_k^{(1)}\big)}{J_k\big(\hat{x}_k^{(2)}\big)}

where :math:`LCR_k` denotes the log-cost ratio between the two hypotheses, :math:`J_k` the total remaining squared error
of the models, and :math:`\hat{x}_k^{(1)}` or :math:`\hat{x}_k^{(2)}` the state estimate when fitting the models
applying the first or second assumption, respectively.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import lmlib as lm
from lmlib.utils.generator import gen_rand_ramps, gen_wgn

# ------------ signal generation -------------

K = 4000  # number of samples (length of test signal)
k = np.arange(K)  # indices
y = 0.5 * gen_rand_ramps(K, N=6, seed=5000) + gen_wgn(K, sigma=0.01, seed=1000)

# --------------- main -----------------------

# Defining ALSSM models
alssm_line_left = lm.AlssmPoly(Q=1, label="line-model-left-sided")
alssm_line_right = lm.AlssmPoly(Q=1, label="line-model-right-sided")

# Defining segments with a left- resp. right-sided decaying window
segment_left = lm.Segment(a=-100, b=-1, direction=lm.FORWARD, g=50)
segment_right = lm.Segment(a=0, b=100, direction=lm.BACKWARD, g=50)

# Defining the final cost function (a so called composite cost = CCost)
# mapping matrix F: maps models to segments (rows = models, columns = segments)
F = [[1, 0], [0, 1]]
ccosts = lm.CCost((alssm_line_left, alssm_line_right), (segment_left, segment_right), F)

# filter signal
se_param = lm.SEParam()
se_param.filter(ccosts, y)

"""two-line approach: cost minimization between the signal and the two models: the two models are forced to be 
continuous at evaluation index k, i.e., the lines are connected but of individual slope """
# connecting lines offset of the two models in sample k
# defining linear constraints to the model states: offset of the two line models at index k is forced to be equal
H1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
xs_1 = se_param.minimize_lin(H1)

"""single-line approach: cost minimization between the signal and the two models: the two models are forced to be 
continuous and derivable at evaluation index k, i.e., the lines are connected AND of common slope """
# connecting lines offset and slope of the two models in sample k defining linear constraints to the model states:
# offset AND slope of the two line models at index k is forced to be equal
H2 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
xs_2 = se_param.minimize_lin(H2)

J1 = se_param.eval_se(xs_1, ref=(k, K))  # get SE of two-line approach
J2 = se_param.eval_se(xs_2, ref=(k, K))  # get SE of single-line approach

# Calculate log-cost ratio between the two approaches
lcr = -0.5 * np.log(J1 / J2)

# find peaks
peaks, _ = find_peaks(lcr, height=0.05, distance=50)

# --------------- plotting of results -----------------------

ks = peaks
trajs_1 = ccosts.trajectory(xs_1[ks], F=F, ref=(ks, K), merge_k="max-win", thd=0.01)
trajs_2 = ccosts.trajectory(xs_2[ks], F=F, ref=(ks, K), merge_k="max-win", thd=0.01)


y_hat = ccosts.eval(xs_1, f=[1, 0], ref=(k, K))
fig, axs = plt.subplots(3, 1, sharex="all")

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)

axs[0].plot(k, y, color="gray", lw=0.25, label="$y$")
axs[0].plot(peaks, y[peaks], ".", color="r", markersize=8, markeredgewidth=1)
axs[0].plot(k, trajs_1, c="blue", lw=1.3)
axs[0].legend(loc="upper right")

axs[1].plot(k, lcr, label="LCR (log-cost ratio)")
axs[1].plot(peaks, lcr[peaks], "x", color="b", markersize=8, markeredgewidth=1.5)
axs[1].legend(loc="lower right")

axs[2].plot(k, J1, lw=1.0, color="blue", label=r"$J(\hat{x}_k^{(1)})$ (no edge)")
axs[2].plot(k, J2, lw=1.0, color="green", label=r"$J(\hat{x}_k^{(2)})$ )(with edge)")
axs[2].legend(loc="lower right")
axs[2].set(xlabel="time index $k$")
plt.show()
