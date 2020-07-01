"""
Script that locally fits the a polynomial of degree 3 to the LCR with an exponential window.

Author: Gustavo Cid Ornelas, ETH Zurich, October 2019
"""
import numpy as np
import matplotlib.pyplot as plt

from window_models import *

if __name__ == "__main__":
    # loading the LCR
    LCR = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/analysis/decaying_sinusoid_0.9_gamma_0.997_Az_"
        "90_freq_80.0.csv",
        delimiter=",",
    )
    # getting rid of the first 500 samples, due to window effect
    LCR = LCR[100000:, :]
    LCR_L = LCR[:, 1]  # LCR for the signal from the left
    LCR_R = LCR[:, 2]  # LCR for the signal from the right
    LCR = np.vstack((LCR_L, LCR_R))
    LCR = LCR.transpose()

    L = LCR.shape[0]  # numb er of samples
    M = LCR.shape[1]  # number of channels (treating the LCR as a multi-channel signal)

    # loading the 3rd degree polynomial model
    n_states = 4
    A_poly = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    s_poly = np.array([[0], [0], [0], [1]])
    # initializing the product of the state transition matrix and the initial state
    As = np.dot(A_poly, s_poly)

    # loading the window model (for now, only exponential)
    n_window = 1
    window_decay = 0.9992
    window = WindowModels(n_window, window_decay)
    window.exponential_window_lssm()

    # pre-computing the Kronecker products used in the message passing
    A_poly_window = np.kron(A_poly, window.A)
    s_poly_window = np.kron(s_poly, window.s)

    # initializing the messages
    chi_k = np.zeros((n_window, 1))  # c_k
    zeta_k = np.zeros((n_window * n_states, M))  # X_k
    s_k = np.zeros((n_window * n_states, n_states))
    w_k = np.zeros((n_states, n_states))  # W_k
    xi_k = np.zeros((n_states, M))  # E_k
    k_k = 0  # K_k

    # arrays that store the fits (LCRs from the left and right)
    poly_fit = np.zeros((L, 2))

    # message passing loop
    for k in range(L):
        if k % 1000 == 0:
            print("Signal sample " + str(k) + "/" + str(L))

        # current signal samples
        LCR_k = LCR[k, :].reshape(M, 1)

        # updating the messages
        chi_k = np.dot(window.A, chi_k) + window.s * (
            np.power(LCR_k[0, 0], 2) + np.power(LCR_k[1, 0], 2)
        )
        temp_chi_k_L = np.dot(window.A, chi_k) + window.s * np.power(
            LCR_k[0, 0], 2
        )  # just for the left channel
        temp_chi_k_R = np.dot(window.A, chi_k) + window.s * np.power(
            LCR_k[1, 0], 2
        )  # just for the right channel
        zeta_k = np.dot(A_poly_window, zeta_k) + np.dot(
            s_poly_window, np.transpose(LCR_k)
        )
        s_k = np.linalg.multi_dot([A_poly_window, s_k, np.transpose(A_poly)]) + np.dot(
            s_poly_window, np.transpose(s_poly)
        )
        # transforming to normal messages
        k_k = np.dot(window.C, chi_k)
        temp_k_k_L = np.dot(window.C, temp_chi_k_L)  # just for the left channel
        temp_k_k_R = np.dot(window.C, temp_chi_k_R)  # just for the right channel

        for row in range(n_states):
            for col_xi in range(M):
                mask_xi = np.zeros((M, n_states))
                mask_xi[col_xi, row] = 1
                xi_k[row, col_xi] = np.trace(np.dot(np.kron(mask_xi, window.C), zeta_k))

            for col_w in range(n_states):
                mask_w = np.zeros((n_states, n_states))
                mask_w[col_w, row] = 1
                w_k[row, col_w] = np.trace(np.dot(np.kron(mask_w, window.C), s_k))

        # to avoid numerical instability
        if (
            abs(k_k) < 1e-16
            or abs(
                k_k
                - np.trace(
                    np.linalg.multi_dot([np.transpose(xi_k), np.linalg.pinv(w_k), xi_k])
                )
            )
            < 1e-16
        ):
            k_k = k_k + 1e-10

        # determining the output matrix for the onset model
        C_poly = np.transpose(np.dot(np.linalg.pinv(w_k), xi_k))

        # calculating the fit
        poly_fit[k, :] = np.dot(C_poly, As).reshape(M)

        # updating state
        As = np.dot(A_poly, As)

    # visualizing the results
    fig, axs = plt.subplots(2)
    axs[0].plot(range(L), LCR_L, "r")
    axs[0].plot(range(L), LCR_R, "b")
    axs[1].plot(range(L), poly_fit[:, 0], "r")
    axs[1].plot(range(L), poly_fit[:, 1], "b")
    plt.show()
