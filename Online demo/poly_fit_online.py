"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""

import numpy as np
import matplotlib.pyplot as plt


class FitPolyOnline:
    """
    Class that contains the method to perform the polynomial fit to the LCR.

    Parameters:
    ----------
    frequencies (list): frequencies of the onset models used to generate the LCRs that will be fit with the polynomial
    LCR_file (string): string with the full path to the csv file containing the LCRs
    azimuth (float): float that represents the value of the azimuth of the sound source (used only to save the file with
                  the correct name)
    """
    def __init__(self):
        # creating the polynomial state space model
        self.n_states = 4
        self.A_poly = np.array([[1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 1],
                                [0, 0, 0, 1]])
        self.A_poly_inv = np.linalg.inv(self.A_poly)
        self.s_poly = np.array([[0],
                                [0],
                                [0],
                                [1]])

        # creating the rectangular window
        self.n_window = 1
        self.window_decay = 1
        self.length_window = 201
        self.a = int((1 - self.length_window) / 2)  # beginning of the window
        self.b = int((self.length_window - 1) / 2)  # end of the window

        # some useful products used in the recursions
        self.A_a_s = np.dot(np.linalg.matrix_power(self.A_poly, self.a), self.s_poly)
        self.A_b_1_s = np.dot(np.linalg.matrix_power(self.A_poly, self.b + 1), self.s_poly)

    def initialize_messages(self, LCR_window_L, k_k_f, k_k_b, xi_k_f, xi_k_b, W_k_f, W_k_b):
        """
        LCR_L is an array of the same length as the length of the window. In this part, we're putting the window over
        in its first position and computing the initial values of the messages
        :param LCR_L:
        :param k_k_f:
        :param k_k_b:
        :param xi_k_f:
        :param xi_k_b:
        :param W_k_f:
        :param W_k_b:
        :return:
        """
        # starting at the initial position
        # forward pass
        for k in range(self.a, 0):
            k_k_f += np.power(LCR_window_L[k - self.a], 2)
            xi_k_f += np.dot(np.linalg.matrix_power(self.A_poly, k), self.s_poly) * LCR_window_L[k - self.a]
            W_k_f += np.linalg.multi_dot([np.linalg.matrix_power(self.A_poly_inv, -k), self.s_poly, np.transpose(
                self.s_poly), np.transpose(np.linalg.matrix_power(self.A_poly_inv, -k))])

        # backward pass
        for k in range(0, self.b + 1):
            k_k_b += np.power(LCR_window_L[k + self.b], 2)
            xi_k_b += np.dot(np.linalg.matrix_power(self.A_poly, k), self.s_poly) * LCR_window_L[k + self.b]
            W_k_b += np.linalg.multi_dot([np.linalg.matrix_power(self.A_poly, k), self.s_poly, np.transpose(
                self.s_poly), np.transpose(np.linalg.matrix_power(self.A_poly, k))])

        # combining the two messages
        W_k = W_k_f + W_k_b
        xi_k = xi_k_f + xi_k_b
        self.W_k_inv = np.linalg.inv(W_k)

        # calculating the optimal C
        C_opt = np.transpose(np.dot(self.W_k_inv, xi_k))
        C_opt = np.dot(C_opt, np.array([[0, 1 / 3.0, -0.5, 1 / 6.0],
                                        [0, -0.5, 0.5, 0],
                                        [0, 1, 0, 0],
                                        [1, 0, 0, 0]]))

        return k_k_f, k_k_b, xi_k_f, xi_k_b, W_k_f, W_k_b, C_opt, LCR_window_L

    def fit_polynomial(self, LCR_window_L, LCR_new_L, k_k_f, k_k_b, xi_k_f, xi_k_b, W_k_f, W_k_b):
        """
        Method that locally fits the a polynomial of degree 3 to the LCR using a rectangular window. The method uses the
        class attributes and the index parameter and then saves a csv file with the corresponding coefficients of the
        polynomial fit.

        Parameter:
        ----------
        index (int): int that represents the column from the LCR file that is used to fit the polynomial, i.e., selects
                if the polynomial is fit to the LCR from the right or from the left. index = 1, selects the LCR from
                the left. index = 2, selects the LCR from the right
        """

        # updating the messages
        # forward pass
        k_k_f = k_k_f - np.power(LCR_window_L[0], 2) + np.power(LCR_window_L[int(len(LCR_window_L) / 2)], 2) ######
        xi_k_f = np.dot(self.A_poly_inv, xi_k_f - self.A_a_s * LCR_window_L[0] + self.s_poly *
                        LCR_window_L[int(len(LCR_window_L) / 2)])

        # backward pass
        k_k_b = k_k_b - np.power(LCR_window_L[int(len(LCR_window_L) / 2)], 2) + np.power(LCR_new_L, 2)
        xi_k_b = np.dot(self.A_poly_inv, xi_k_b - self.s_poly * LCR_window_L[int(len(LCR_window_L) / 2)] +
                        self.A_b_1_s * LCR_new_L)

        # combining the two messages
        xi_k = xi_k_f + xi_k_b
        k_k = k_k_f + k_k_b

        # calculating the optimal C
        C = np.transpose(np.dot(self.W_k_inv, xi_k))

        C_opt = np.dot(C, np.array([[0, 1 / 3.0, -0.5, 1 / 6.0],
                                    [0, -0.5, 0.5, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0]]))

        # samples of the LCR inside the window
        LCR_window_L = np.hstack((LCR_window_L[1:], LCR_new_L))

        return k_k_f, k_k_b, xi_k_f, xi_k_b, W_k_f, W_k_b, C_opt, LCR_window_L

