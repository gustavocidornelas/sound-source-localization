"""
.. _lmlib_statespace_eval_functions:

Evaluation Functions
====================

Here we explain the different eval functions of the state space module and how to use them in all different ways.


"""


import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm

# import alssm_eval_functions_plot


def plot_example_eval_at_ref(alssm, xs, js, ax):
    ys = alssm.eval_at(xs, js)
    for n, y in enumerate(ys):
        ax.plot(js, y, "--k", lw=0.5)
        ax.annotate(
            "$p_{}(k)$".format(n),
            (js[-3], y[-3]),
            xytext=(js[-3] + 1, y[-3]),
            arrowprops=dict(arrowstyle="-"),
        )


def plot_example_eval(js, ys, ax):
    for j, y, n in zip(js, ys, np.arange(len(ys))):
        ax.scatter([j], [y])
        ax.annotate(
            "$s_0(x_{})$".format(n),
            (j, y),
            xytext=(j - 2, y + 3),
            arrowprops=dict(arrowstyle="-"),
        )
    ax.set_xlabel("$k$")


# plot_example_eval_at()
# plot_example_eval_ref()


#######################################
# Pre-Script
# ----------
# Creating an polynomial autonomous linear state space model with system order N=3.
#

alssm = lm.AlssmPoly(Q=2)
xs = [[2, 1, 0.5]]

# for example plot
fig, ax = plt.subplots(1, 1)
ax.set_title("Reference polynomial $p_0(k)$")
plot_example_eval_at_ref(alssm, xs, js=np.arange(-10, 11), ax=ax)
plt.show()

###################################
# Evaluation of an ALSSM with a single state vector
# -------------------------------------------------
# Such a evaluation of an ALSSM is achieved with the method :meth:`~lmlib.statespace.model.Alssm.eval()`.
#
# :math:`y = CA^0x_0 = Cx_0`
#

ys = alssm.eval(xs)

# for example plot
fig, ax = plt.subplots(1, 1)
plot_example_eval_at_ref(alssm, xs, js=np.arange(-10, 11), ax=ax)
plot_example_eval(js=[0], ys=ys, ax=ax)
plt.show()

###################################
# Multiple Evaluations of an ALSSM with independent state vectors
# ---------------------------------------------------------------
#
# :math:`y_i = CA^0 x_i = Cx_i`
#

xs = [[10, 0, -0.2], [0, 2, 0.3], [-5, 0.3, 0.1]]
ys = alssm.eval(xs)

###############################
# Generates plot


fig, ax = plt.subplots(1, 1)
plot_example_eval_at_ref(alssm, xs, js=np.arange(-10, 11), ax=ax)
plot_example_eval(js=[0, 0, 0], ys=ys, ax=ax)
plt.show()


###################################
# Evaluations of an ALSSM at a discrete time
# ------------------------------------------
#
# :math:`y_j= CA^jx_0`
#
