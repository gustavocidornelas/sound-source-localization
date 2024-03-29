"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


class DelayEstimation:
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
    signal_id (string): identifier of the audio signal (used only to save the file with the correct name)
    delay_saving_path (string): full path indicating where the file with the delays should be saved
    """

    def __init__(
        self,
        coeff_left_file,
        coeff_right_file,
        frequencies,
        azimuth,
        delay_saving_path,
        signal_id,
    ):
        self.coeff_left_file = coeff_left_file
        self.coeff_right_file = coeff_right_file
        self.frequencies = frequencies
        self.azimuth = azimuth
        self.delay_saving_path = delay_saving_path
        self.signal_id = signal_id

    def _get_delay(self, roots):
        """
        Method that estimates the delay between the two signals based on the roots found on the second degree-polynomial
        approximation
    
        Parameter:
        -----------
        roots (np.array): array with the roots found when estimating the shift between the signals
    
        Returns:
        ----------
        delay (float): delay estimate
        """
        if roots[0] > 0 and roots[1] > 0:
            delay = np.min(roots)
        elif roots[0] < 0 and roots[1] > 0:
            delay = roots[1]
        elif roots[0] > 0 and roots[1] < 0:
            delay = roots[0]

        return delay

    def estimate_delay(self):
        """
        Method that  estimates the delay between two sound signals based on the local polynomial approximation of their
        corresponding LCRs. The method uses the class attributes and saves a csv file with the unique delays.
        """
        # loading the files with the coefficients of the corresponding signals LCRs (in the canonical basis)
        coeff_left_file = self.coeff_left_file
        coeff_right_file = self.coeff_right_file
        azimuth = self.azimuth
        frequencies = self.frequencies

        print("Estimating the delays...")

        coeff_left = np.genfromtxt(coeff_left_file, delimiter=",", skip_header=False)
        coeff_right = np.genfromtxt(coeff_right_file, delimiter=",", skip_header=False)

        # some useful computations and parameters
        # number of samples in sequence used to do the continuous delay estimation
        N = 50
        # window size for the 2nd degree polynomial approximation
        a0 = -40.0
        b0 = 40.0
        A = np.array([[1, 0, 0], [1, a0, np.power(a0, 2)], [1, b0, np.power(b0, 2)]])
        A_inv = np.linalg.inv(A)

        # array that stores the estimated delays
        total_delay = np.zeros(coeff_left.shape[0])
        delay = 0
        count = 0
        continuous_delay_status = (
            0  # used to check if it should perform continuous delay estimation or not
        )

        # main loop, going through each sample coefficients
        for k in range(coeff_left.shape[0]):
            # retrieving the current sample's coefficients
            coeff_left_k = coeff_left[k, :]
            coeff_right_k = coeff_right[k, :]

            # checking if it is a good sample to estimate the delay (both increasing signals)
            if (
                coeff_left_k[1] > 1e-6
                and 2 * coeff_left_k[2] < 1e-13
                and coeff_right_k[1] > 1e-6
                and 2 * coeff_right_k[2] < 1e-13
                or continuous_delay_status == 1
            ):
                print(k)
                # approximating the 3rd degree polynomials by 2nd degree polynomials (coefficients beta) on a smaller window
                beta_left = np.dot(
                    A_inv,
                    np.array(
                        [
                            [coeff_left_k[0]],
                            [
                                coeff_left_k[0]
                                + coeff_left_k[1] * a0
                                + coeff_left_k[2] * np.power(a0, 2)
                                + coeff_left_k[3] * np.power(a0, 3)
                            ],
                            [
                                coeff_left_k[0]
                                + coeff_left_k[1] * b0
                                + coeff_left_k[2] * np.power(b0, 2)
                                + coeff_left_k[3] * np.power(b0, 3)
                            ],
                        ]
                    ),
                )
                beta_right = np.dot(
                    A_inv,
                    np.array(
                        [
                            [coeff_right_k[0]],
                            [
                                coeff_right_k[0]
                                + coeff_right_k[1] * a0
                                + coeff_right_k[2] * np.power(a0, 2)
                                + coeff_right_k[3] * np.power(a0, 3)
                            ],
                            [
                                coeff_right_k[0]
                                + coeff_right_k[1] * b0
                                + coeff_right_k[2] * np.power(b0, 2)
                                + coeff_right_k[3] * np.power(b0, 3)
                            ],
                        ]
                    ),
                )

                # checking which 2nd degree polynomial is in front
                if beta_left[0, 0] > beta_right[0, 0]:
                    # solving the system to find when the signal in the right reaches the same level as the one in the left
                    roots = np.roots(
                        np.array(
                            [
                                beta_right[2, 0],
                                beta_right[1, 0],
                                beta_right[0, 0] - beta_left[0, 0],
                            ]
                        )
                    )

                    # estimating the delay based on the roots
                    if np.isreal(roots).all():
                        # delay = get_delay(roots)
                        if roots[0] > 0 and roots[1] > 0:
                            delay = np.min(roots)
                        elif roots[0] < 0 and roots[1] > 0:
                            delay = roots[1]
                        elif roots[0] > 0 and roots[1] < 0:
                            delay = roots[0]
                        # count = 0
                    # print(delay)

                else:
                    # solving the system to find when the signal in the left reaches the same level as the one in the right
                    roots = np.roots(
                        np.array(
                            [
                                beta_left[2, 0],
                                beta_left[1, 0],
                                beta_left[0, 0] - beta_right[0, 0],
                            ]
                        )
                    )

                    # estimating the delay based on the roots
                    if np.isreal(roots).all():
                        # delay = get_delay(roots)
                        if roots[0] > 0 and roots[1] > 0:
                            delay = np.min(roots)
                        elif roots[0] < 0 and roots[1] > 0:
                            delay = roots[1]
                        elif roots[0] > 0 and roots[1] < 0:
                            delay = roots[0]
                        # count = 0
                    print(delay)

                # setting the continuous delay estimation status to 1 if the first onset is detected and to 0 after the
                # continuous delay estimation period
                if count < N:
                    continuous_delay_status = 1
                    count = count + 1
                elif count == N:
                    continuous_delay_status = 0
                    count = 0
                    # computing the mean of the delays estimated during the continuous delay estimation period
                    delay = np.mean(total_delay[k - N : k])
                    total_delay[k - N : k] = np.median(total_delay[k - N : k])

            total_delay[k] = delay

        np.save(
            self.delay_saving_path
            + "/"
            + self.signal_id
            + "_all_delays_Az_"
            + str(azimuth)
            + ".npy",
            total_delay,
        )
        # print(np.unique(total_delay)/44.1)
        # print(np.median(np.unique(total_delay)/44.1))
        # np.savetxt('unique_delays_Az_' + str(azimuth) + '_freq_' + str(frequencies) + '.csv', np.unique(total_delay),
        #           delimiter=',')
        # plt.figure()
        # plt.plot(np.asarray(range(total_delay.shape[0]))/44100, total_delay/44.1, linewidth=1.8)
        # plt.ylabel('ITD [ms]')
        # plt.xlabel('Time [s]')
        # plt.grid(ls='--', c='.5')

        # plt.show()
