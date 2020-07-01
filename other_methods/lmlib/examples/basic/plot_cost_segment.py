"""
.. _lmlib_example_cost_segment:

Cost Segment
============

This example defines a cost segment using a discrete-time autonomous linear state space model (ALSSM) and a left-sided,
exponentially decaying window. See: :class:`~lmlib.statespace.costfunc.Segment`, :class:`~lmlib.statespace.costfunc.CostSegment`,
:class:`~lmlib.statespace.model.Alssm`.
"""
import numpy as np
import lmlib as lm

# Defining an second order polynomial ALSSM
Q = 2
alssm_poly = lm.AlssmPoly(Q, label="alssm-polynomial")

# Defining a segment with a left-sided, exponentially decaying window
a = -np.inf  # left boundary
b = -1  # right boundary
g = 50  # effective weighted number of sample under the window (controlling the window size)
left_seg = lm.Segment(a, b, lm.FORWARD, g, label="left-decaying")

# creating the cost segment, combining window (segment) and model (ALSSM).
costs = lm.CostSegment(alssm_poly, left_seg, label="costs segment for polynomial model")

# print internal structure
print(costs)
