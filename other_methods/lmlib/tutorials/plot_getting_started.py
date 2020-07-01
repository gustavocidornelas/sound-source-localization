#!/usr/bin/env python
# coding: utf-8
"""
Getting Started
***************
"""
##############################
# This is a beginner's tutorial to start using the model-based signal processing library lmlib.
#
# Description
# -----------
# This tutorial creates the lmlib emblem step by step.
# To do so, we locally approximate an electrocardiogram (ECG) signal sequence with a polynomial model.
# This includes the following steps:
#
# * loading ECG signal sniped from example signal library;
# * setting-up an autonomous linear state space model (ALSSM) generating an output sequence of polynomial shape;
# * setting-up a segment which defines the interval borders and the shape of the window used when the polynomial is fit
#   to the ECG data in the next steps;
# * defining the squared error cost function to be minimized when doing the model fit by combining the previously
#   defined ALSSM with the segment configuration;
# * minimizing the just defined cost function over the initial state of the ALSSM.
#
# Following these steps, we will gain our emblem.
#
#
# Imports
# .......
# This tutorial requires numpy, matplotlib, and this lmlib package.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import lmlib as lm

from lmlib.utils.generator import load_single_channel

##############################
# Loading an example signal
# .........................
#
# We use an exemplary electrocardiogram (ECG) signal, provided by lmlib;
# the function ``lmlib.ecg_signal_single_chanel()`` returns such an ECG sniped in a 1-dimensional numpy array of 24000
# samples.
y = load_single_channel("EECG_BASELINE_1CH_10S_FS2400HZ.csv", K=-1)
K = len(y)  # read number of samples
k = np.arange(K)  # generate discrete-time index array

# plotting the signal
fig, ax = plt.subplots(figsize=(7, 1.7))
ax.plot(k, y)
plt.show()

##############################
# Setting up an ALSSM model
# -------------------------
# We now set up an ALSSM model.
# An ALSSM is a discrete-time linear state space model (LSSM) with no input.
# It follows that, once a single state is fixed, all future and past states of the model are fixed.
# An ALSSM model can produce a whole plethora of outputs, whereof common are *polynomial*, *sinusoidal*, and
# *exponential* shapes, including linear combinations and multiplications of those.
#
# The output sequence of an ALSSM is given, for an initial state :math:`x_0`,
#
# .. math::
#    s_i(x_0) = C A^ix_0 \ ,
#
# where :math:`s_i(x_k) \in \mathbb{R}` is the output at index :math:`i`,
# :math:`C \in \mathbb{R}^{L \times N}` the output matrix,
# :math:`A \in \mathbb{R}^{N \times N}` the state-transition matrix, and
# :math:`x_k \in \mathbb{R}^N` is the state space vector (independent variable).
#
# More general, if the model is moved to (localized around) any time index :math:`k`,
# we get
#
# .. math::
#    s_i(x_k) = C A^ix_k \ ,
#
# where index :math:`k` refers to the absolute time axis of the signal, and
# index :math:`i` is on a local time axis with respect to index :math:`k`.
# The following image illustrates the two axis.
#
# .. image:: /_static/time_domains.png
#    :width: 600
#    :align: center
#    :alt: time domains
#
# However, for our example, a single-channel 3th order polynomial, :math:`L=1`  and :math:`N=4`,
# with
#
# .. math::
#    A &= \begin{bmatrix}1 & 1 & 1 & 1\\ 0 & 1 & 2 & 3 \\ 0 & 0 & 1 & 3\\ 0 & 0 &0 &1\end{bmatrix}\\
#    C &= \begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}
#
# is used:
#


alssm_poly = lm.AlssmPoly(Q=3)
alssm_poly.update()
print(alssm_poly)

##############################
#  To update all internal structures,
# ``alssm_poly.update()`` needs to be called before any usage of a model.
#
# Setting up segments
# -------------------
# Now, we set up a segment.
# A segment provides localization to a ALSSM, i.e. it adds a weighted window of finite or infinite borders.
# A segment consists of interval borders and a window shape,
# weighting the single samples in the cost function accordingly when the ECG signal is fit to the model samples.
# The boundaries :math:`a` and :math:`b` and can be declared as finite or infinite (at least on of needs to be finite).
# The window weights as an additional number, giving the number of weighted samples in a window, and is defined
# as `effective sample weight` :math:`g\in \mathbb{R}_+`. Because of the recursive cost function we need to
# define a computation direction.
#
# Lmlib's advantage is to combine diverse segments with models to achieve high modular signal processing procedures
# quite efficient. More on this in the next section.
#
# Now we generate two segments  - a left-sided segment where the window weight lower as we go left
# and a right-sided segment where the window weight lowe as we go right.

seg_left = lm.Segment(a=-300, b=0, direction=lm.FORWARD, g=20)
seg_right = lm.Segment(a=1, b=300, direction=lm.BACKWARD, g=20)

##############################
# Lets plot the window weights of the two segments above the ECG-signal. As we are plotting the signal is in the
# global time-domain, the segment needs a reference. The reference index `k0` sets
# start of the models time-domain.

k0 = [11200]  # reference index global

# getting the window weights
win_left = seg_left.window(ref=(k0, K))
win_right = seg_right.window(ref=(k0, K))

# plotting segments and signal
_, axs = plt.subplots(2, 1, sharex="all")
axs[0].plot(k, win_left, label="left-sided segment")
axs[0].plot(k, win_right, label="right-sided segment")
axs[1].plot(k, y)
axs[0].legend()

# show just a interval of the plot
plt.xlim(k0[0] - 600, k0[0] + 300)
plt.show()


##############################
# Assign Model to Segment
# -----------------------
# Before we setup the cost function we need to map the model to the segment.
# In this case with just one model its quite simple. Systems with multiple models and segments increase the varieties.
#
# The assignment is done with a matrix :math:`F`,
# where the first dimension corresponds to the models and the second to the segments.
# All models in a single row will be stacked to one large model with summed outputs, where the output matrices are
# scaled with the row-entries of :math:`F`.
#
# For example:
# If we have 2 models and 3 segments and the mapping matrix
#
# .. math::
#    F = \begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} \ ,
#
# where the columns refer to the models and the row to the segments.
# It means that the first model is assigned to the center segment and the second model to all segments.
#
# Build Mapping matrix
# ....................
# We assign the model to both segment with this :math:`F` matrix:

F = [[1, 1]]

#############################################################
# Setup the Cost Function
# -----------------------
# The cost function is given for each index :math:`k`. where :math:`x_k` is the independent variable.
# The squared error between the model output and the signal over a segment is defined as *cost segment*
#
# .. math::
#    J_a^b(k,x_k) = \sum_{j=k+a}^{k+b} w_{j-k}(y_j - CA^{j-k}x_k)^2
#
# This cost segment is for one model and one segment. So we use the *composite cost*
#
# .. math::
#    J_P(k, x_k) = \sum_{p=1}^P J_{a_p}^{b_p} (k,x_k)
#
# that defines the cost over multiple models and segments.
#
c_cost = lm.CCost((alssm_poly,), (seg_left, seg_right), F)


###############################
# Filter Parameter
# ----------------
# The filter parameters are varialbes of the expandend squared error cost function.
# They result from the forward and backward recursion. See [Wildhaber2018]_ [Eq. 27-34].
# Beside the recursions, the class :class:`~lmlib.statepace.fparam.FParam` allocates memory for the filter parameters.
# We create at first the object, and then calculate the parameters with the
# :meth:`~lmlib.statepace.fparam.FParam.filter` method.

se_param = lm.SEParam()
se_param.filter(c_cost, y)

##################################
# Minimize the composite costs
# ----------------------------
# Now we can minimize the cost function with the least square approche by calling
# :meth:`~lmlib.statepace.fparam.FParam.minimize()`.
# The method returns an state space vector estimate :math:`\hat{x}`.
#
xs = se_param.minimize()

##################################
# Plot the trajectories
# ---------------------
# First we evaluate the at global time index k0 the model with the state vector estimate :math:`\hat{x}` at k0,
# then we mask this output, with the window mask.
# At last we plot the signals.

trajs = c_cost.trajectory(xs[k0], F=F, ref=(k0, K), thd=1e-2, merge_k="max-win")
trajs_long = c_cost.trajectory(xs[k0], F=F, ref=(k0, K), thd=1e-5, merge_k="max-win")

_, axs = plt.subplots(2, 1, sharex="all")
axs[0].plot(k, win_left, label="left-sided segment")
axs[0].plot(k, win_right, label="right-sided segment")
axs[1].plot(k, y, c="grey", lw=0.3)
axs[1].plot(k, trajs, lw=1.5)
axs[1].plot(k, trajs_long, lw=1, ls=":")
plt.xlim(k0[0] - 600, k0[0] + 300)
plt.show()


##################################
# Designing The lmlib-Logo
# ------------------------

fig, ax = plt.subplots(figsize=(8, 8))

ax.axvline(k0, ls="-", c="k", lw=2)
ax.axhline(y[k0], ls="-", c="k", lw=2)

ax.plot(k, y, c="k", lw=4.5)

ax.plot(k, trajs_long, lw=3, c="w", ls="--")
ax.plot(k, trajs, lw=5.5, c="r", ls="-")
ax.axis("off")

plt.xlim(k0[0] - 300, k0[0] + 300)
plt.ylim(y[k0] - 0.1, y[k0] + 0.1)
plt.savefig("logo.svg", transparent=True)
plt.show()
