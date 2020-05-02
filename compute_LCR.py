"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, September 2019
"""

import numpy as np
import matplotlib.pyplot as plt

from onset_models import *
from window_models import *


class ComputeLCR:
    """
    Class that contains the method fits the specified onset model to the audio data.

    Parameters:
    ----------
    frequencies (list): list with the frequencies (float) used. If a single frequency is used, then it should be a list
                    with a single element
    n_freq (int): number of frequencies being used in the model (i.e., for the linear combination scenario). If fitting
                one onset model at a time, n_freq should be equal to one
    onset_decays (list): list with the onset decays used (float) used. It must match the length of f and n_freq
    window_decay (float): float that represents the window decay
    audio_file (string): string with the full path to the csv file containing the audio signals
    azimuth (float): float that represents the value of the azimuth of the sound source (used only to save the file with
                  the correct name)
    onset_model (string): onset model used. Can be either 'decaying_sinusoid' or 'gammatone'
    window_model (string): window model used. Can be either 'exponential' or 'gamma'
    signal_id (string): identifier of the audio signal (used only to save the file with the correct name)
    lcr_saving_path (string): full path indicating where the LCR file should be saved
    """
    def __init__(self, onset_model, window_model, frequencies, n_freq, onset_decays, window_decay, audio_file, azimuth,
                 signal_id, lcr_saving_path):
        self.frequencies = frequencies
        self.n_freq = n_freq
        self.onset_decays = onset_decays
        self.window_decay = window_decay
        self.audio_file = audio_file
        self.azimuth = azimuth
        self.onset_model = onset_model
        self.window_model = window_model
        self.signal_id = signal_id
        self.lcr_saving_path = lcr_saving_path

    def compute_lcr(self):
        """
        Method that fits the onset model to the audio signal. It uses the class attributes and saves a csv file with the
        LCRs for both channels (left and right).
        """
        # reading the parameters
        f = self.frequencies
        n_freq = self.n_freq
        onset_decay = self.onset_decays
        window_decay = self.window_decay
        audio_file = self.audio_file
        azimuth = self.azimuth
        onset_model = self.onset_model
        window_model = self.window_model

        print('Computing the LCRs...')

        # checking for consistency with the parameters
        assert len(f) == len(onset_decay), 'Number of frequencies and decays do not match.'
        assert len(f) == n_freq, 'Length of frequency list does not match the number of frequencies specified.'

        # loading the audio file
        # audio_data = np.genfromtxt(audio_file, delimiter=',')  # when the audio file is a .csv
        audio_data = np.load(audio_file)  # when the audio file is a .npy

        y_L = audio_data[:, 0]  # signal from the microphone in the left
        y_R = audio_data[:, 1]  # signal from the microphone in the right

        y = audio_data[:, [0, 1]]  # multi-channel signal

        #plt.figure()
        #plt.plot(np.asarray(range(y_L.shape[0])), y_L)
        #plt.plot(np.asarray(range(y_L.shape[0])), y_R)
        #plt.show()

        L = y.shape[0]  # number of samples
        M = y.shape[1]  # number of channels (one for the right and one for the left)

        # onset model
        model = OnsetModels(f, onset_decay, n_freq)
        if onset_model == 'decaying_sinusoid':
            # number of states for the decaying sinusoid
            n_states = 2 * n_freq
            # loading the decaying sinusoid model
            model.decaying_sinusoid_lssm(n_states)

        elif onset_model == 'gammatone':
            # number of states for the gammatone
            n_states = 8 * n_freq
            # loading the gammatone model
            model.gammatone_lssm(n_states)

            # P matrix (for the gammatone case only)
            P = np.kron(np.eye(n_freq, dtype=int), np.hstack((np.eye(2), np.zeros((2, 6)))))

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
        chi_k = np.zeros((n_window, 1))
        zeta_k = np.zeros((n_window * n_states, M))
        s_k = np.zeros((n_window * n_states, n_states))
        w_k = np.zeros((n_states, n_states))
        xi_k = np.zeros((n_states, M))
        k_k = 0

        # stores the minimum cost
        min_J = np.zeros(L)
        min_J_L = np.zeros(L)
        min_J_R = np.zeros(L)
        LCR = np.zeros((L, 1))
        LCR_L = np.zeros((L, 1))
        LCR_R = np.zeros((L, 1))

        # message passing loop
        for k in range(L):
            print(k)
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

            # avoid having a singular matrix in the first iterations
            # if k <= 1:
            #    w_k = w_k + np.random.rand(n_states, n_states) * 1e-14

            if onset_model == 'decaying_sinusoid':
                # determining the output matrix for the onset model
                C_onset = np.transpose(np.dot(np.linalg.pinv(w_k), xi_k))

                # calculating the minimum cost
                min_J_L[k] = temp_k_k_L - np.linalg.multi_dot(
                    [np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]])
                min_J_R[k] = temp_k_k_R - np.linalg.multi_dot(
                    [np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]])

                min_J[k] = k_k - np.trace(np.linalg.multi_dot([np.transpose(xi_k), np.linalg.pinv(w_k), xi_k]))

            elif onset_model == 'gammatone':
                # determining the output matrix for the onset model
                PWPt_inv = np.linalg.pinv(np.linalg.multi_dot([P, w_k, np.transpose(P)]))
                C_onset = np.dot(np.transpose(np.linalg.multi_dot([PWPt_inv, P, xi_k])), P)

                min_J_L[k] = temp_k_k_L - np.linalg.multi_dot(
                    [np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]])
                min_J_R[k] = temp_k_k_R - np.linalg.multi_dot(
                    [np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]])

                # calculating the minimum cost
                min_J[k] = k_k - np.trace(np.linalg.multi_dot([np.transpose(xi_k), np.transpose(P), PWPt_inv, P, xi_k]))

            # calculating the local cost ratio (LCR)
            LCR[k] = - (1 / 2) * np.log10(min_J[k] / k_k)
            LCR_L[k] = - (1 / 2) * np.log10(min_J_L[k] / temp_k_k_L)
            LCR_R[k] = - (1 / 2) * np.log10(min_J_R[k] / temp_k_k_R)

        # saving the results
        final_LCR = np.hstack((LCR, LCR_L, LCR_R))
        np.savetxt(self.lcr_saving_path + "/" + self.signal_id + "_" + str(onset_model) + "_" + str(onset_decay) + "_"
                   + str(window_model) + "_" + str(window_decay) + "_Az_" + str(azimuth) + "_freq_" + str(f) + ".csv",
                   final_LCR, delimiter=",", header="Multi-channel, Left, Right")


