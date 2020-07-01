"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, September 2019
"""

import numpy as np
import matplotlib.pyplot as plt

from onset_models import *
from window_models import *


class ComputeLCROnline:
    """
    Class that contains the method fits the specified onset model to the audio in an online manner.

    Parameters:
    ----------
    frequencies (list): list with the frequencies (float) used. If a single frequency is used, then it should be a list
                    with a single element
    n_freq (int): number of frequencies being used in the model (i.e., for the linear combination scenario). If fitting
                one onset model at a time, n_freq should be equal to one
    onset_decays (list): list with the onset decays used (float) used. It must match the length of f and n_freq
    window_decay (float): float that represents the window decay
    onset_model (string): onset model used. Can be either 'decaying_sinusoid' or 'gammatone'
    window_model (string): window model used. Can be either 'exponential' or 'gamma'
    """

    def __init__(
        self, onset_model, window_model, frequencies, n_freq, onset_decays, window_decay
    ):
        self.frequencies = frequencies
        self.n_freq = n_freq
        self.onset_decays = onset_decays
        self.window_decay = window_decay
        self.onset_model = onset_model
        self.window_model = window_model

        # checking if the attributes are consistent
        assert len(self.frequencies) == len(
            self.onset_decays
        ), "Number of frequencies and decays do not match."
        assert len(self.frequencies) == self.n_freq, (
            "Length of frequency list does not match the number of "
            "frequencies specified."
        )

        # creating the onset model
        self.model = OnsetModels(self.frequencies, self.onset_decays, self.n_freq)
        if self.onset_model == "decaying_sinusoid":
            # number of states for the decaying sinusoid
            self.n_states = 2 * self.n_freq
            # loading the decaying sinusoid model
            self.model.decaying_sinusoid_lssm(self.n_states)

        elif self.onset_model == "gammatone":
            # number of states for the gammatone
            self.n_states = 8 * self.n_freq
            # loading the gammatone model
            self.model.gammatone_lssm(n_states)

            # P matrix (for the gammatone case only)
            self.P = np.kron(
                np.eye(self.n_freq, dtype=int), np.hstack((np.eye(2), np.zeros((2, 6))))
            )

        # creating the window model
        if self.window_model == "gamma":
            self.n_window = 4
            self.window = WindowModels(self.n_window, self.window_decay)
            self.window.gamma_window_lssm()
        elif self.window_model == "exponential":
            self.n_window = 1
            self.window = WindowModels(self.n_window, self.window_decay)
            self.window.exponential_window_lssm()

        # pre-computing the Kronecker products used in the message passing
        self.A_model_window = np.kron(self.model.A, self.window.A)
        self.s_model_window = np.kron(self.model.s, self.window.s)

    def compute_lcr(self, y, chi_k, zeta_k, s_k, w_k, xi_k, k_k):
        """
        Method that fits the onset model to the audio signal. It uses the class attributes and saves a csv file with the
        LCRs for both channels (left and right).
        """
        y_L = y[0]  # signal from the microphone in the left
        y_R = y[1]  # signal from the microphone in the right

        M = len(y)  # number of channels (one for the right and one for the left)

        # current signal samples
        y_k = y.reshape(M, 1)

        # updating the messages
        chi_k = np.dot(self.window.A, chi_k) + self.window.s * (
            np.power(y_k[0, 0], 2) + np.power(y_k[1, 0], 2)
        )
        temp_chi_k_L = np.dot(self.window.A, chi_k) + self.window.s * np.power(
            y_k[0, 0], 2
        )  # just for the left channel
        temp_chi_k_R = np.dot(self.window.A, chi_k) + self.window.s * np.power(
            y_k[1, 0], 2
        )  # just for the right channel
        zeta_k = np.dot(self.A_model_window, zeta_k) + np.dot(
            self.s_model_window, np.transpose(y_k)
        )
        s_k = np.linalg.multi_dot(
            [self.A_model_window, s_k, np.transpose(self.model.A)]
        ) + np.dot(self.s_model_window, np.transpose(self.model.s))
        # transforming to normal messages
        k_k = np.dot(self.window.C, chi_k)
        temp_k_k_L = np.dot(self.window.C, temp_chi_k_L)  # just for the left channel
        temp_k_k_R = np.dot(self.window.C, temp_chi_k_R)  # just for the right channel

        for row in range(self.n_states):
            for col_xi in range(M):
                mask_xi = np.zeros((M, self.n_states))
                mask_xi[col_xi, row] = 1
                xi_k[row, col_xi] = np.trace(
                    np.dot(np.kron(mask_xi, self.window.C), zeta_k)
                )

            for col_w in range(self.n_states):
                mask_w = np.zeros((self.n_states, self.n_states))
                mask_w[col_w, row] = 1
                w_k[row, col_w] = np.trace(np.dot(np.kron(mask_w, self.window.C), s_k))

        # avoid having a singular matrix in the first iterations
        # if k <= 1:
        #    w_k = w_k + np.random.rand(n_states, n_states) * 1e-14

        if self.onset_model == "decaying_sinusoid":
            # determining the output matrix for the onset model
            C_onset = np.transpose(np.dot(np.linalg.pinv(w_k), xi_k))

            # calculating the minimum cost
            min_J_L = temp_k_k_L - np.linalg.multi_dot(
                [np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]]
            )
            min_J_R = temp_k_k_R - np.linalg.multi_dot(
                [np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]]
            )

            min_J = k_k - np.trace(
                np.linalg.multi_dot([np.transpose(xi_k), np.linalg.pinv(w_k), xi_k])
            )

        elif self.onset_model == "gammatone":
            # determining the output matrix for the onset model
            PWPt_inv = np.linalg.pinv(
                np.linalg.multi_dot([self.P, w_k, np.transpose(self.P)])
            )
            C_onset = np.dot(
                np.transpose(np.linalg.multi_dot([PWPt_inv, self.P, xi_k])), self.P
            )

            min_J_L = temp_k_k_L - np.linalg.multi_dot(
                [np.transpose(xi_k[:, 0]), np.linalg.pinv(w_k), xi_k[:, 0]]
            )
            min_J_R = temp_k_k_R - np.linalg.multi_dot(
                [np.transpose(xi_k[:, 1]), np.linalg.pinv(w_k), xi_k[:, 1]]
            )

            # calculating the minimum cost
            min_J = k_k - np.trace(
                np.linalg.multi_dot(
                    [np.transpose(xi_k), np.transpose(self.P), PWPt_inv, self.P, xi_k]
                )
            )

        # calculating the local cost ratio (LCR)
        LCR_L = -(1 / 2) * np.log10(min_J_L / temp_k_k_L)
        LCR_R = -(1 / 2) * np.log10(min_J_R / temp_k_k_R)

        LCR = np.hstack((LCR_L, LCR_R))

        return chi_k, zeta_k, s_k, w_k, xi_k, k_k, LCR
