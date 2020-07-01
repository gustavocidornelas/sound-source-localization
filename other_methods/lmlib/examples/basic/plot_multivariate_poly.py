# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
"""
Multivariate Polynomial
=======================

This example generates a multivariate polynomial using the :class:`PolyMulti` class. The polynomial is shown in a 3-D
surface plot. plot.

"""

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np

# creating a multivariate polynomial with two variables of order 2
pm1 = lm.MPoly(coefs=([0.1, -0.3, -0.1, 0.1],), expos=([0, 2], [0, 2]))

# defining the variables for evaluation
x = np.arange(-3, 3, 0.1)
y = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(x, y)

z = pm1.eval((X, Y))


# create figure and an axes for a 3d plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# mesh gird and reshape z

# surface plot of the multivariate polynomial
ax.plot_surface(X, Y, z, cmap="twilight")
ax.set(xlabel="x", ylabel="y", zlabel=r"$p(x,y)$", title="Multivariate Polynomial")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.show()
