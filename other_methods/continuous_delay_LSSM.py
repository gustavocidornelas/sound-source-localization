"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import lmlib as lm
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # window for the delay estimate
    a = -30
    b = 30

    # alpha is the set of coefficients for the left
    alpha = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/Other methods/signal_left.csv",
        delimiter=",",
        skip_header=False,
    )
    # beta is the set of coefficients for the right
    beta = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/Other methods/signal_right.csv",
        delimiter=",",
        skip_header=False,
    )

    # list with the exponents
    q = [0, 1, 2, 3]

    # pre-computations
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

    # delay interval
    s = np.arange(-100, 100, 1)

    ITD = []

    # loop going through the coefficients
    for k in range(a, alpha.shape[0] - b):
        # retrieving the current sample's coefficients
        alpha_k = alpha[k, :]
        beta_k = beta[k, :]

        # observation
        observation_coef = (
            A @ np.kron(alpha_k, alpha_k)
            - B @ np.kron(alpha_k, beta_k)
            + C @ np.kron(beta_k, beta_k)
        )

        # squared Error
        J = lm.Poly(observation_coef, q2)
        js = J.eval(s)

        # computing the delay
        delay_id = np.argmin(js)
        ITD.append(s[delay_id] / 44.1)

    # smoothed version (moving average)
    smooth_delay = np.convolve(np.array(ITD), np.ones(50) / 50, mode="same")

    audio = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_30.csv",
        delimiter=",",
    )
    audio_onset = audio[238000:238600, :]

    fig, axs = plt.subplots(2)
    axs[0].set_title("Continuous delay estimation for the azimuth 30 degrees")
    axs[0].plot(
        range(audio_onset.shape[0]), audio_onset[:, 0], label="Left", linewidth=2
    )
    axs[0].plot(
        range(audio_onset.shape[0]), audio_onset[:, 1], label="Right", linewidth=2
    )
    # axs[0].set_xlim(0, 350)
    axs[1].plot(range(len(ITD)), ITD, label="Continuous estimate", linewidth=2)
    axs[1].plot(
        range(smooth_delay.shape[0]), smooth_delay, label="Moving average", linewidth=2
    )
    axs[1].plot(
        range(smooth_delay.shape[0]),
        -0.221 * np.ones(smooth_delay.shape[0]),
        "k--",
        label="Theoretical delay",
    )
    # axs[1].set_ylim(0, 1)
    # axs[1].set_xlim(0, 350)
    axs[1].set_ylabel("Delay [ms]")
    axs[1].set_xlabel("k")
    axs[0].legend()
    axs[1].legend()
    plt.show()
