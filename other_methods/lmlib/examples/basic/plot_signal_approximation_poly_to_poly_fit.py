"""
Signal Approximation by polynomial ALSSM and low-order-polynomial fit
=====================================================================

"""

import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
from lmlib.utils.generator import gen_sinusoidal, gen_rand_walk

# Generate a Signal
K = 1000  # signal length
k = np.arange(K)  # sample index vector
y = gen_sinusoidal(K, k_period=400) + 0.1 * gen_rand_walk(
    K, seed=1000
)  # sinusoidal with additive random walk

###################
# Defining a Model

# Polynomial ALSSM
Q = 5  # LSSM System Order
alssm_poly_Q = lm.AlssmPoly(Q - 1)  # Q-1 polynomial degree

# Segment (Exponential Window starting at a ends at b-1, the decay is given by g (area) direction defines the
# computation direction)
segment_right = lm.Segment(a=0, b=100, direction=lm.BACKWARD, g=80)

# Cost Segment connects ALSSM model with Segment
costs_Q = lm.CostSegment(alssm_poly_Q, segment_right)

# filter signal and take the approximation
se_param = lm.SEParam()  # data storage object
se_param.filter(costs_Q, y)  # filter data with the cost model defined above
xs = (
    se_param.minimize()
)  # minimize the costs using squared error filter parameters, xs are the polynomial coefficients

#########################
# Polynomial to Polynomial Approximation

# constant calculation
a = segment_right.a  # left boundary a has to be finite
b = segment_right.b  # right boundary b has to be finite
q = np.arange(Q)  # exponent vector from 0 to Q-1

R = 2  # Polynomial order (degree +1) of the polynomial approximation ## This is the lower order polynomial
r = np.arange(R)  # exponent vector from 0 to R-1

# Constant Calculation (See. Signal Analysis Using Local Polynomial
# Approximations. Appendix)
s = np.concatenate([q, r])
A = np.concatenate([np.eye(Q), np.zeros((R, Q))], axis=0)
B = np.concatenate([np.zeros((Q, R)), np.eye(R)], axis=0)
Ms = lm.poly_square_expo(s)
L = lm.poly_int_coef_L(Ms)
c = np.power(b, Ms + 1) - np.power(a, Ms + 1)
vec_C = L.T @ c
C = vec_C.reshape(np.sqrt(len(vec_C)).astype(int), -1)

# Solves the Equation by setting derivative to zero.
Lambda = np.linalg.inv(2 * B.T @ C @ B) @ (2 * B.T @ C @ A)
betas = np.einsum(
    "ij, nj -> ni", Lambda, xs
)  # approximated low order polynomial coefficients

# ----------------  Plot  -----------------
ks = [150, 600]  # show trajectory and windows at the indices in ks
wins = costs_Q.window(ref=(ks, K))  # windows

trajs_Q = costs_Q.trajectory(
    xs[ks], ref=(ks, K), thd=0.1
)  # trajectories of the higher order polynomial

costs_R = lm.CostSegment(lm.AlssmPoly(R - 1), segment_right)
trajs_R = costs_R.trajectory(
    betas[ks], ref=(ks, K), thd=0.1
)  # trajectories of the lower order polynomial

y_hat = costs_Q.alssm.eval(xs[k], ref=(ks, K))  # signal estimate

# Generate the plot
fig, axs = plt.subplots(2, 1, sharex="all")
axs[0].plot(k, wins, lw=1, c="k", ls="-")
axs[0].set_ylabel("window weights")
axs[0].legend([f"windows at {ks}"])
axs[1].plot(k, y, lw=0.5, c="grey")
axs[1].plot(k, y_hat, lw=0.1, c="k")
axs[1].plot(k, trajs_Q, lw=1.5)
axs[1].plot(k, trajs_R, "--", lw=1.5)
axs[1].set(xlabel="k", ylabel="amplitude")
axs[1].legend(
    [
        "$y$",
        "$\\hat{y}$",
        f"poly Q at {ks[0]}",
        f"poly Q at {ks[1]}",
        f"poly R at {ks[0]}",
        f"poly R at {ks[1]}",
    ]
)

plt.show()
