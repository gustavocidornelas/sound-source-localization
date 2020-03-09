"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


class DelayEstimationOnline:
    """
    Class that contains the methods used to perform the delay estimation (not in the filter bank scenario).

    Parameters:
    ----------
    coeff_left_file (string): string with the full path to the csv file containing the polynomial coefficients for the
                            left LCR
    coeff_right_file (string): string with the full path to the csv file containing the polynomial coefficients for the
                           right LCR
    frequencies (list): frequencies of the onset models used to generate the LCRs that will be fit with the polynomial
    azimuth (float): float that represents the value of the azimuth of the sound source (used only to save the file with
                  the correct name)
    """

    def __init__(self):
        # some useful computations and parameters
        # window size for the 2nd degree polynomial approximation
        self.a0 = -40.0
        self.b0 = 40.0
        self.A = np.array([[1, 0, 0],
                          [1, self.a0, np.power(self.a0, 2)],
                          [1, self.b0, np.power(self.b0, 2)]])
        self.A_inv = np.linalg.inv(self.A)

    def estimate_delay(self, coeff_left_k, coeff_right_k, count, delay):
        """
        Method that  estimates the delay between two sound signals based on the local polynomial approximation of their
        corresponding LCRs. The method uses the class attributes and saves a csv file with the unique delays.
        """

        # checking if it is a good sample to estimate the delay (both increasing signals)
        if (coeff_left_k[1] > 1e-6 and 2 * coeff_left_k[2] < 1e-13 and
                coeff_right_k[1] > 1e-6 and 2 * coeff_right_k[2] < 1e-13 and count > 1000):
            count = 0
            # approximating the 3rd degree polynomials by 2nd degree polynomials (coefficients beta) on a smaller window
            beta_left = np.dot(self.A_inv, np.array([[coeff_left_k[0]],
                                                     [coeff_left_k[0] + coeff_left_k[1] * self.a0 +
                                                      coeff_left_k[2] * np.power(self.a0, 2) + coeff_left_k[3] *
                                                      np.power(self.a0, 3)],
                                                    [coeff_left_k[0] + coeff_left_k[1] * self.b0 +
                                                     coeff_left_k[2] * np.power(self.b0, 2) + coeff_left_k[3] *
                                                     np.power(self.b0, 3)]]))
            beta_right = np.dot(self.A_inv, np.array([[coeff_right_k[0]],
                                                      [coeff_right_k[0] + coeff_right_k[1] * self.a0 +
                                                       coeff_right_k[2] * np.power(self.a0, 2) + coeff_right_k[3] *
                                                       np.power(self.a0, 3)],
                                                      [coeff_right_k[0] + coeff_right_k[1] * self.b0 +
                                                       coeff_right_k[2] * np.power(self.b0, 2) + coeff_right_k[3] *
                                                       np.power(self.b0, 3)]]))

            # checking which 2nd degree polynomial is in front
            if beta_left[0, 0] > beta_right[0, 0]:
                # solving the system to find when the signal in the right reaches the same level as the one in the left
                roots = np.roots(np.array([beta_right[2, 0], beta_right[1, 0], beta_right[0, 0] - beta_left[0, 0]]))

                # estimating the delay based on the roots
                if np.isreal(roots).all():
                    # delay = get_delay(roots)
                    if roots[0] > 0 and roots[1] > 0:
                        delay = np.min(roots)
                    elif roots[0] < 0 and roots[1] > 0:
                        delay = roots[1]
                    elif roots[0] > 0 and roots[1] < 0:
                        delay = roots[0]

            else:
                # solving the system to find when the signal in the left reaches the same level as the one in the right
                roots = np.roots(np.array([beta_left[2, 0], beta_left[1, 0], beta_left[0, 0] - beta_right[0, 0]]))

                # estimating the delay based on the roots
                if np.isreal(roots).all():
                    # delay = get_delay(roots)
                    if roots[0] > 0 and roots[1] > 0:
                        delay = np.min(roots)
                    elif roots[0] < 0 and roots[1] > 0:
                        delay = roots[1]
                    elif roots[0] > 0 and roots[1] < 0:
                        delay = roots[0]
        return delay, count



