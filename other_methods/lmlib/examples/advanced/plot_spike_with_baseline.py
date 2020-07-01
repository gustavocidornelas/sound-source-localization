# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
r"""
Example: Rectangular Pulse with Baseline
========================================

This example demonstrates the detection of a rectangular pulses of known duration in a noisy signal with baseline
interferences.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import lmlib as lm
from lmlib.utils.generator import (
    gen_rand_pulse,
    gen_baseline_sin,
    gen_wgn,
    gen_rand_walk,
)

# --------------- parameters of example -----------------------
K = 4000  # number of samples (length of test signal)
k = np.arange(K)
len_pulse = 20  # [samples] number of samples of the pulse width
y_rpulse = 0.03 * gen_rand_pulse(K, N=6, length=len_pulse, seed=1000)
y = y_rpulse + gen_baseline_sin(K, k_period=2000) + gen_wgn(K, sigma=0.01, seed=1000)

LCR_THD = 0.1  # minimum log-cost ratio to detect a pulse in noise

g_sp = 15000  # pulse window weight, effective sample number under the window # (larger value lead to a more rectangular-like windows while too large values might lead to nummerical instabilities in the recursive computations.)
g_bl = 50  # baseline window weight, effective sample number under the window (larger value leads to a wider window)

# --------------- main -----------------------

# Defining ALSSM models
alssm_pulse = lm.AlssmPoly(Q=0, label="line-model-pulse")
alssm_baseline = lm.AlssmPoly(Q=2, label="offset-model-baseline")

# Defining segments with a left- resp. right-sided decaying window and a center segment with nearly rectangular window
segmentL = lm.Segment(a=-np.inf, b=-1, g=g_bl, direction=lm.FORWARD)
segmentC = lm.Segment(a=0, b=len_pulse, g=g_sp, direction=lm.FORWARD)
segmentR = lm.Segment(
    a=len_pulse + 1, b=np.inf, g=g_bl, direction=lm.BACKWARD, delta=len_pulse
)

# Defining the final cost function (a so called composite cost = CCost)
# mapping matrix between models and segments (rows = models, columns = segments)
F = [[0, 1, 0], [1, 1, 1]]
ccosts = lm.CCost((alssm_pulse, alssm_baseline), (segmentL, segmentC, segmentR), F)


# filter signal
se_param = lm.SEParam()
se_param.filter(ccosts, y)
xs = se_param.minimize()
y_hat = ccosts.eval(xs, f=[1, 0], ref=(k, K))

xs_0 = np.copy(xs)
xs_0[:, 0] = 0

k = np.arange(K)
J = se_param.eval_se(
    xs, ref=(k, K)
)  # get SE (squared error) for hypothesis 1 (baseline + pulse)
J0 = se_param.eval_se(
    xs_0, ref=(k, K)
)  # get SE (squared error)  for hypothesis 0 (baseline only) --> J0 should be a vector not a matrice

LCR = -0.5 * np.log(J / J0)

# find peaks
peaks, _ = find_peaks(LCR, height=LCR_THD, distance=30)


# --------------- plotting of results -----------------------

fig, axs = plt.subplots(4, 1, sharex="all")

# Remove horizontal space between axes
fig.subplots_adjust(hspace=0.1)

if peaks.size != 0:
    wins = ccosts.window(
        ps=[1, 1, 1], ref=(peaks, K), merge_seg="none", merge_k="max-win"
    )
    axs[0].plot(k, wins, color="r", lw=0.25, ls="-", label=r"$\alpha_k(.)$")
axs[0].set(ylabel="windows")

axs[1].plot(k, y, color="grey", lw=0.25, label="y")
axs[1].plot(k, y_rpulse, color="black", lw=0.5, linestyle="-", label="pulses")
axs[1].plot(
    peaks, y[peaks], "s", color="r", fillstyle="none", markersize=8, markeredgewidth=1.0
)
axs[1].legend(loc="upper right")

axs[2].plot(k, LCR, lw=1.0, color="green", label=r"$LCR = J(\hat{\lambda}_k) / J(0)$")
axs[2].plot(peaks, LCR[peaks], "x", color="r", markersize=8, markeredgewidth=2.0)
axs[2].axhline(LCR_THD, color="black", linestyle="--", lw=1.0)
axs[2].legend(loc="upper right")

axs[3].plot(k, y_hat, lw=1.0, color="gray", label=r"$\hat{\lambda}_{k}$")
# _ , stemlines, _ = axs[3].stem(peaks, x[peaks, 0], markerfmt="bo", basefmt=" ")
# plt.setp(stemlines,'linewidth', 3)
axs[3].axhline(0, color="black", lw=0.5)
axs[3].legend(loc="upper right")
axs[3].set(xlabel="time index $k$")

plt.show()
