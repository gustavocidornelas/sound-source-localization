"""
Modified Bryson–Frazier smoother
================================

Discrete-time signal smoother using the modified Bryson-Frazier message passing schema, see  “On Sparsity by NUV-EM,
Gaussian Message Passing, and Kalman Smoothing” [Loeliger2016]_ . Implemented in
:meth:`~lmlib.statespace.smoother.mbf_smoother_d`.

"""
import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_triangle, gen_rand_walk, gen_rand_pulse

#  Generating test signal
K = 1000
k = np.arange(K)
y = gen_triangle(K, 2000) + 0.05 * gen_rand_walk(K, seed=1000)

# Create Integrator LSSM
A = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
B = [1, 0, 0]
C = [0, 0, 1]
D = 0

lssm = lm.Lssm(A, B, C, D, label="integrator")
lssm.update()

# Modified Bryson–Frazier smoother
mx, Vx = lm.mbf_smoother_d(lssm, sigma2U=1e-1, sigma2Z=1e6, y=y)

# Evaluate
y_hat = lssm.eval(mx)
sigma2Z_hat = lssm.C @ Vx @ lssm.C.T

# Plot
fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]}, sharex="all")
axs[0].plot(k, y, lw=0.6, c="gray", label="$y$")
axs[0].plot(k, y_hat, lw=1, c="r", label="$\\hat{y}$", ls="--")
axs[0].legend(loc="upper right")
axs[0].set_ylabel("Amplitude")

axs[1].plot(k, sigma2Z_hat, label="$\\hat{\\sigma}_Z^2$")
axs[1].legend(loc="upper right")
axs[1].set_ylabel("Variance")
axs[1].set_xlabel("time index $k$")
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[1].set_yscale("log")
plt.show()
