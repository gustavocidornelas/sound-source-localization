"""
Minimum Squared Error Fit of Polynomials by Scaling in its Amplitude and its Time
=================================================================================

"""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import matplotlib.pyplot as plt
import lmlib as lm

# parameters
q = [0, 1, 2, 3]

alpha = np.array([1, 0.2, 1.5, 0.2])
beta = np.array([-3.5, -0.5, 2.5, 1.1])
L = lm.poly_dilation_coef_L(q, 2)
beta = 7 * np.dot(L, alpha)
a = -1
b = 1


# constant calculation
Mq = lm.poly_square_expo(q)
Mq1 = lm.poly_int_expo(Mq)
L_int = lm.poly_int_coef_L(Mq)
L_int_ab = lm.mpoly_int_coef_L((q, Mq), position=1)
L_int_bb = lm.mpoly_int_coef_L((Mq, Mq), position=1)
R_QQ = lm.permutation_matrix_square(len(q), len(q))
DELTA_Q = lm.mpoly_dilate_ind_coef_L(q)

c1 = L_int
c2 = L_int_ab
c3 = L_int_bb @ R_QQ

A = np.dot(np.transpose(np.power(b, Mq1) - np.power(a, Mq1)), c1)
B = np.dot(
    np.kron(np.eye(len(q)), np.transpose(np.power(b, Mq1) - np.power(a, Mq1))), c2
) @ np.kron(DELTA_Q, np.eye(len(q)))
C = np.dot(
    np.kron(np.eye(len(q) ** 2), np.transpose(np.power(b, Mq1) - np.power(a, Mq1))), c3
) @ np.kron(DELTA_Q, DELTA_Q)


lam = 0.5
eta = 0.5
J = (
    np.dot(A, np.kron(alpha, alpha))
    - 2 * lam * np.dot(B @ np.kron(beta, alpha), np.power(eta, q))
    + lam ** 2 * np.dot(C @ np.kron(beta, beta), np.power(eta, Mq))
)

p1 = lm.Poly(alpha, q)
p2 = lm.Poly(beta, q)

x = 2
print(p1.eval(x) - np.dot(lam, p2.eval(eta * x)))
print(
    np.dot(alpha, np.power(x, q))
    - np.dot(
        lam, np.dot(np.dot(DELTA_Q, beta), np.kron(np.power(eta, q), np.power(x, q)))
    )
)


print((p1.eval(x) - np.dot(lam, p2.eval(eta * x))) ** 2)
print(
    np.dot(np.kron(alpha, alpha), np.power(x, Mq))
    - np.dot(
        2 * lam,
        np.dot(
            np.kron(DELTA_Q @ beta, alpha), np.kron(np.power(eta, q), np.power(x, Mq))
        ),
    )
    + np.dot(
        lam ** 2,
        np.dot(
            R_QQ @ np.kron(DELTA_Q @ beta, DELTA_Q @ beta),
            np.kron(np.power(eta, Mq), np.power(x, Mq)),
        ),
    )
)


print((p1.eval(x) - np.dot(lam, p2.eval(eta * x))) ** 2)
print(
    np.dot(np.kron(alpha, alpha), np.power(x, Mq))
    - np.dot(
        2 * lam,
        np.dot(
            np.kron(DELTA_Q, np.eye(len(q))) @ np.kron(beta, alpha),
            np.kron(np.power(eta, q), np.power(x, Mq)),
        ),
    )
    + np.dot(
        lam ** 2,
        np.dot(
            R_QQ @ np.kron(DELTA_Q, DELTA_Q) @ np.kron(beta, beta),
            np.kron(np.power(eta, Mq), np.power(x, Mq)),
        ),
    )
)


print((p1.eval(x) - np.dot(lam, p2.eval(eta * x))) ** 2)
print(
    np.dot(np.kron(alpha, alpha), np.power(x, Mq))
    - np.dot(
        2 * lam,
        np.dot(
            np.kron(DELTA_Q, np.eye(len(q))) @ np.kron(beta, alpha),
            np.kron(np.power(eta, q), np.power(x, Mq)),
        ),
    )
    + np.dot(
        lam ** 2,
        np.dot(
            R_QQ @ np.kron(DELTA_Q, DELTA_Q) @ np.kron(beta, beta),
            np.kron(np.power(eta, Mq), np.power(x, Mq)),
        ),
    )
)

A = np.transpose(np.power(b, Mq1) - np.power(a, Mq1)) @ lm.poly_int_coef_L(
    Mq
)  # poly_integral_undef_c
B = lm.mpoly_def_int_coef_L((q, Mq), 1, a, b)
C = lm.mpoly_def_int_coef_L((Mq, Mq), 1, a, b)

print((p1.eval(x) - np.dot(lam, p2.eval(eta * x))) ** 2)
print(
    np.dot(A, np.kron(alpha, alpha))
    - np.dot(
        2 * lam,
        np.dot(
            B @ np.kron(DELTA_Q, np.eye(len(q))) @ np.kron(beta, alpha),
            np.power(eta, q),
        ),
    )
    + np.dot(
        lam ** 2,
        np.dot(
            C @ R_QQ @ np.kron(DELTA_Q, DELTA_Q) @ np.kron(beta, beta),
            np.power(eta, Mq),
        ),
    )
)


B = B @ np.kron(DELTA_Q, np.eye(len(q)))
C = C @ R_QQ @ np.kron(DELTA_Q, DELTA_Q)
# Squared Error
J = 0


p1 = lm.poly_square(lm.Poly(B @ np.kron(beta, alpha), q))
p2 = lm.Poly(C @ np.kron(beta, beta), Mq)
p1_diff = lm.poly_diff(p1)
p2_diff = lm.poly_diff(p2)
p2_square = lm.poly_square(p2)
p1dp2 = lm.poly_prod((p1_diff, p2))
p1p2d = lm.poly_prod((p1, p2_diff))
p1p2d.scale(-1)
# (num, denum) = poly_q_diff(p_num, p_denum)


step = 0.01
x0 = 0
delta_error = 1e-6
x_old = x0
x_new = x_old
error = np.inf
while error >= delta_error:
    x_old = x_new
    x_new = x_old + step * (p1dp2.eval(x_old) + p1p2d.eval(x_old)) * 1 / (
        p2_square.eval(x_old)
    )
    error = abs(x_old - x_new)

lam_hat = lm.Poly(B @ np.kron(beta, alpha), q).eval(x_old) / p2.eval((x_old))
print(x_old, lam_hat, error)
# ------------------ plot ------------------

x = np.arange(-4, 4.01, 0.01)
p1 = lm.Poly(alpha, q)
p2 = lm.Poly(beta, q)

y1 = p1.eval(x)
y2 = p2.eval(x)

fig, axs = plt.subplots(2, 1)
axs[0].plot(x, y1, "r--", label="$p_{\\alpha}$")
axs[0].plot(x, y2, "g--", label="$p_{\\beta}$")
axs[0].fill_between(
    x,
    y1,
    y2,
    where=np.bitwise_and(x < b, x >= a),
    color=(0.9, 0.9, 0.9),
    label="$J(\\lambda = 1, \\eta = 1)",
)
axs[0].axvline(a, c="k", lw=0.5)
axs[0].axvline(b, c="k", lw=0.5)
axs[0].legend()
axs[0].set_xlabel("$x$")
axs[0].set_title(
    "Minimum Squared Error Fit of Polynomials by Scaling in its Amplitude and its Time"
)


lam = 0.5
etas = np.arange(-20, 20, 0.5)
Js = [
    np.dot(A, np.kron(alpha, alpha))
    - 2 * lam * np.dot(B @ np.kron(beta, alpha), np.power(eta, q))
    + lam ** 2 * np.dot(C @ np.kron(beta, beta), np.power(eta, Mq))
    for eta in etas
]
axs[1].plot(etas, Js, "r--", label="$p_{\\alpha}$")

plt.show()


# defining the variables for evaluation
x = np.arange(-2, 2, 0.01)
y = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(x, y)

lams = y
etas = x

Js = [
    [
        np.dot(A, np.kron(alpha, alpha))
        - 2 * lam * np.dot(B @ np.kron(beta, alpha), np.power(eta, q))
        + lam ** 2 * np.dot(C @ np.kron(beta, beta), np.power(eta, Mq))
        for eta in etas
    ]
    for lam in lams
]

Js = np.asarray(Js)
# create figure and an axes for a 3d plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# mesh gird and reshape z

# surface plot of the multivariate polynomial
ax.plot_surface(X, Y, np.log(Js), cmap="viridis")
ax.set(xlabel="x", ylabel="y", zlabel=r"$p(x,y)$", title="Multivariate Polynomial")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.show()
