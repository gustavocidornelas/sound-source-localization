"""
Script that locally fits the a polynomial of degree 3 to the LCR using a rectangular window.

Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    # retrieving the parameters
    LCR_file = str(sys.argv[1])
    index = int(sys.argv[2])
    azimuth = float(sys.argv[3])

    # loading the LCR
    LCR = np.genfromtxt(LCR_file, delimiter=',')
    # getting rid of the first 500 samples, due to window effect
    LCR = LCR[500:, :]
    LCR_L = LCR[:, index]  # LCR for the signal from the left
    LCR_R = LCR[:, 2]  # LCR for the signal from the right
    LCR = np.vstack((LCR_L, LCR_R))
    LCR = LCR.transpose()

    L = LCR.shape[0]  # number of samples
    M = LCR.shape[1]  # number of channels (treating the LCR as a multi-channel signal)

    # loading the 3rd degree polynomial model
    n_states = 4
    A_poly = np.array([[1, 1, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1]])
    A_poly_inv = np.linalg.inv(A_poly)
    s_poly = np.array([[0],
                       [0],
                       [0],
                       [1]])

    # defining the rectangular window parameters
    n_window = 1
    window_decay = 1
    length_window = 101  # 201
    a = int((1 - length_window) / 2)  # beginning of the window
    b = int((length_window - 1) / 2)  # end of the window

    # initializing the other messages
    k_k_f = 0
    k_k_b = 0
    xi_k_f = np.zeros((n_states, 1))
    xi_k_b = np.zeros((n_states, 1))
    W_k_f = np.zeros((n_states, n_states))
    W_k_b = np.zeros((n_states, n_states))

    # sliding the rectangular window through the signal
    # starting at the initial position
    # forward pass
    for k in range(a, 0):
        k_k_f += np.power(LCR_L[k - a], 2)
        xi_k_f += np.dot(np.linalg.matrix_power(A_poly, k), s_poly) * LCR_L[k - a]
        W_k_f += np.linalg.multi_dot([np.linalg.matrix_power(A_poly_inv, -k), s_poly, np.transpose(s_poly),
                                      np.transpose(np.linalg.matrix_power(A_poly_inv, -k))])

    # backward pass
    for k in range(0, b + 1):
        k_k_b += np.power(LCR_L[k + b], 2)
        xi_k_b += np.dot(np.linalg.matrix_power(A_poly, k), s_poly) * LCR_L[k + b]
        W_k_b += np.linalg.multi_dot([np.linalg.matrix_power(A_poly, k), s_poly, np.transpose(s_poly),
                                      np.transpose(np.linalg.matrix_power(A_poly, k))])

    # combining the two messages
    W_k = W_k_f + W_k_b
    xi_k = xi_k_f + xi_k_b
    W_k_inv = np.linalg.inv(W_k)
    # calculating the optimal C
    C = np.transpose(np.dot(W_k_inv, xi_k))

    # some useful products used in the recursions
    A_a_s = np.dot(np.linalg.matrix_power(A_poly, a), s_poly)
    A_b_1_s = np.dot(np.linalg.matrix_power(A_poly, b + 1), s_poly)

    # message recursive updates for the other window positions
    C_opt = np.zeros((L + a - 2 + b + 1, 4))
    for k0 in range(b + 1, L + a - 1):

        # forward pass
        k_k_f = k_k_f - np.power(LCR_L[k0-1 + a], 2) + np.power(LCR_L[k0-1], 2)
        xi_k_f = np.dot(A_poly_inv, xi_k_f - A_a_s * LCR_L[k0-1 + a] + s_poly * LCR_L[k0-1])

        # backward pass
        k_k_b = k_k_b - np.power(LCR_L[k0-1], 2) + np.power(LCR_L[k0 + b], 2)
        xi_k_b = np.dot(A_poly_inv, xi_k_b - s_poly * LCR_L[k0-1] + A_b_1_s * LCR_L[k0 + b])

        # combining the two messages
        xi_k = xi_k_f + xi_k_b
        k_k = k_k_f + k_k_b

        # calculating the optimal C
        C = np.transpose(np.dot(W_k_inv, xi_k))

        C_opt[k0, :] = np.dot(C, np.array([[0, 1/3.0, -0.5, 1/6.0],
                                           [0, -0.5, 0.5, 0],
                                           [0, 1, 0, 0],
                                           [1, 0, 0, 0]]))

    if index == 1:
        np.savetxt('coeff_' + str(azimuth) + '_LEFT.csv', C_opt, delimiter=',')
    elif index == 2:
        np.savetxt('coeff_' + str(azimuth) + '_RIGHT.csv', C_opt, delimiter=',')
