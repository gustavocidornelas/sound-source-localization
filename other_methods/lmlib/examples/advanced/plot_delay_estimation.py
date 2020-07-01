"""
Delay Estimation Using Local Polynomial Approximations
======================================================



"""
import lmlib as lm
import matplotlib.pyplot as plt
import numpy as np

a = -2
b = 2
alpha = [2, 0, -2, 0.5]
beta = [2.5, 1, -0.2, 0.1]

q = [0, 1, 2, 3]

Q = len(q)
L = lm.mpoly_shift_coef_L(q)
qts = lm.mpoly_shift_expos(q)
Lt = lm.mpoly_dilate_coef_L(qts, 1, -0.5) @ L
Bt = lm.mpoly_dilate_coef_L(qts, 1, 0.5) @ L
RQQ = lm.permutation_matrix_square(Q, Q)
q2s = lm.mpoly_square_expos(qts)
Ct = lm.mpoly_def_int_coef_L(q2s, 0, a, b) @ RQQ
q2 = lm.mpoly_def_int_expos(q2s, 0)[0]
K = lm.commutation_matrix(len(q2s[0]), len(q2s[1]))
Kt = np.identity(len(q2) ** 2) + K
A = Ct @ np.kron(Lt, Lt)
B = Ct @ Kt @ np.kron(Lt, Bt)
C = Ct @ np.kron(Bt, Bt)

# observation
observation_coef = (
    A @ np.kron(alpha, alpha) - B @ np.kron(alpha, beta) + C @ np.kron(beta, beta)
)

# Squared Error
J = lm.Poly(observation_coef, q2)


# ------------------ plot ------------------


s = np.arange(-4, 4.01, 0.01)
p1 = lm.Poly(alpha, q)
p2 = lm.Poly(beta, q)

y1 = p1.eval(s)
y2 = p2.eval(s)
js = J.eval(s)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(s, y1, "r--", label="$p_{\\alpha}$")
ax1.plot(s, y2, "g--", label="$p_{\\beta}$")
ax1.fill_between(
    s,
    y1,
    y2,
    where=np.bitwise_and(s < b, s >= a),
    color=(0.9, 0.9, 0.9),
    label="$J(\\alpha, \\beta)$",
)
ax1.axvline(a, c="k", lw=0.5)
ax1.axvline(b, c="k", lw=0.5)
ax1.legend()
ax1.set_xlabel("$s$")
ax1.set_title("Two Polynomials and its Squared Error within an Interval")


ax2.plot(s, js)
ax2.set_xlabel("$s$")
ax2.set_ylabel("$J(s)$")
ax2.set_yscale("log")
ax2.set_title("Polynomial Cost Function $J(s)$")

plt.show()
