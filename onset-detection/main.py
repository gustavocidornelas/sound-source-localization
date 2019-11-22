"""
Main script that fits the specified onset model to the audio data. Usage:

python main.py fc_max onset_decay window_decay

Author: Gustavo Cid Ornelas, ETH Zurich, September 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

from parameters import *
from onset_models import *
from window_models import *

if __name__ == '__main__':

    # loading the audio file
    audio_data = np.genfromtxt(audio_file_dir, delimiter=',')
    y_L = audio_data[:, 0]  # signal from the microphone in the left
    y_R = audio_data[:, 2]  # signal from the microphone in the right
    y = audio_data[:, [0, 2]]  # multi-channel signal

    L = y.shape[0]  # number of samples
    M = y.shape[1]  # number of channels (one for the right and one for the left)

    # reading the parameters
    fc_max = float(sys.argv[1])
    onset_decay = float(sys.argv[2])
    window_decay = float(sys.argv[3])

    # define the frequencies for the frequency bank (ERB scale)
    # for now, f = fc_max, defined in the parameter's file, since we work with a single frequency
    ERB_max = 21.4 * np.log10(1 + 0.00437 * fc_max)
    ERB = np.linspace(ERB_max / n_freq, ERB_max, n_freq)
    f = (np.power(10, ERB / 21.4) - 1) / 0.00437

    # onset model
    model = OnsetModels(f, onset_decay)
    if onset_model == 'decaying_sinusoid':
        # loading the decaying sinusoid model
        model.decaying_sinusoid_lssm()
    elif onset_model == 'gammatone':
        # loading the gammatone model
        model.gammatone_lssm()

        # P matrix (for the gammatone case only)
        P = np.hstack((np.eye(2), np.zeros((2, 6))))

    print(model.A)
    # window model
    if window_model == 'gamma':
        n_window = 4
        window = WindowModels(n_window, window_decay)
        window.gamma_window_lssm()
    elif window_model == 'exponential':
        n_window = 1
        window = WindowModels(n_window, window_decay)
        window.exponential_window_lssm()

    # pre-computing the Kronecker products used in the message passing
    A_model_window = np.kron(model.A, window.A)
    s_model_window = np.kron(model.s, window.s)

    # initializing the messages
    chi_k = np.zeros((n_window, 1))  # c_k
    zeta_k = np.zeros((n_window * n_states, M))  # X_k
    s_k = np.zeros((n_window * n_states, n_states))
    w_k = np.zeros((n_states, n_states))  # W_k
    xi_k = np.zeros((n_states, M))  # E_k
    k_k = 0  # K_k

    # stores the minimum cost
    min_J = np.zeros(L)
    min_J_L = np.zeros(L)
    min_J_R = np.zeros(L)
    LCR = np.zeros((L, n_freq))
    LCR_L = np.zeros((L, n_freq))
    LCR_R = np.zeros((L, n_freq))

    # message passing loop
    for k in range(L):
        if k % 1000 == 0:
            print('Signal sample ' + str(k) + '/' + str(L))

        # current signal samples
        y_k = y[k, :].reshape(M, 1)

        # updating the messages
        chi_k = np.dot(window.A, chi_k) + window.s * (np.power(y_k[0, 0], 2) + np.power(y_k[1, 0], 2))
        temp_chi_k_L = np.dot(window.A, chi_k) + window.s * np.power(y_k[0, 0], 2)  # just for the left channel
        temp_chi_k_R = np.dot(window.A, chi_k) + window.s * np.power(y_k[1, 0], 2)  # just for the right channel
        zeta_k = np.dot(A_model_window, zeta_k) + np.dot(s_model_window, np.transpose(y_k))
        s_k = np.linalg.multi_dot([A_model_window, s_k, np.transpose(model.A)]) + np.dot(s_model_window,
                                                                                         np.transpose(model.s))
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
        if abs(k_k) < 1e-16 or abs(k_k - np.trace(np.linalg.multi_dot([np.transpose(xi_k),
                                                                       np.linalg.pinv(w_k), xi_k]))) < 1e-16:
            k_k = k_k + 1e-10

        if onset_model == 'decaying_sinusoid':
            # determining the output matrix for the onset model
            C_onset = np.transpose(np.dot(np.linalg.pinv(w_k), xi_k))

            # calculating the minimum cost
            min_J_L[k] = temp_k_k_L - np.linalg.multi_dot([np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]])
            min_J_R[k] = temp_k_k_R - np.linalg.multi_dot([np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]])

            min_J[k] = k_k - np.trace(np.linalg.multi_dot([np.transpose(xi_k), np.linalg.pinv(w_k), xi_k]))

        elif onset_model == 'gammatone':
            # determining the output matrix for the onset model
            PWPt_inv = np.linalg.pinv(np.linalg.multi_dot([P, w_k, np.transpose(P)]))
            C_onset = np.dot(np.transpose(np.linalg.multi_dot([PWPt_inv, P, xi_k])), P)

            min_J_L[k] = temp_k_k_L - np.linalg.multi_dot([np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]])
            min_J_R[k] = temp_k_k_R - np.linalg.multi_dot([np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]])

            # calculating the minimum cost
            min_J[k] = k_k - np.trace(np.linalg.multi_dot([np.transpose(xi_k), np.transpose(P), PWPt_inv, P, xi_k]))

        # calculating the local cost ratio (LCR)
        LCR[k] = - (1 / 2) * np.log10(min_J[k] / k_k)
        LCR_L[k] = - (1 / 2) * np.log10(min_J_L[k] / temp_k_k_L)
        LCR_R[k] = - (1 / 2) * np.log10(min_J_R[k] / temp_k_k_R)

    # saving the results
    final_LCR = np.hstack((LCR, LCR_L, LCR_R))
    np.savetxt(str(onset_model) + "_" + str(onset_decay) + "_" + str(window_model) + "_" + str(window_decay) + "_Az_" +
               str(audio_file_dir[-6:-4]) + "_freq_" + str(fc_max) + ".csv", final_LCR, delimiter=",",
               header="Multi-channel, Left, Right")

    # visualizing the results
    fig, axs = plt.subplots(3)
    fig.suptitle('')
    axs[0].plot(range(L), y_L)
    axs[1].plot(range(L), y_R)
    axs[2].plot(range(L), LCR_L, 'r')
    axs[2].plot(range(L), LCR_R, 'b')
    plt.show()
