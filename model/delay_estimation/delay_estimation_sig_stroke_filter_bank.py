import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

from scipy.spatial import distance

import lmLib as lm


class DelayEstimationSigStroke:
    """
    Class that contains the method to perform the delay estimation in the case where a filter bank is being used.

    Parameters:
    ----------
    frequencies (list): frequencies of the onset models used to generate the LCRs that will be fit with the polynomial
    azimuth (float): float that represents the value of the azimuth of the sound source (used only to save the file with
                  the correct name)
    signal_id (string): identifier of the audio signal (used only to save the file with the correct name)
    delay_saving_path (string): full path indicating where the file with the delays should be saved
    coefficients_saving_path (string): full path indicating where the files with the polynomial coefficients are saved
    """

    def __init__(
        self,
        frequencies,
        azimuth,
        delay_saving_path,
        coefficients_saving_path,
        signal_id,
    ):
        self.frequencies = frequencies
        self.azimuth = azimuth
        self.delay_saving_path = delay_saving_path
        self.coefficients_saving_path = coefficients_saving_path
        self.signal_id = signal_id

        # pre-computations for the method integrate delay
        # list with the exponents
        self.q = [0, 1, 2, 3]
        # window for the delay estimate
        self.a = -40
        self.b = 40

        # pre-computations
        self.Q = len(self.q)
        self.L = lm.mpoly_shift_coef_L(self.q)
        self.qts = lm.mpoly_shift_expos(self.q)
        self.Lt = lm.mpoly_dilate_coef_L(self.qts, 1, -0.5) @ self.L
        self.Bt = lm.mpoly_dilate_coef_L(self.qts, 1, 0.5) @ self.L
        self.RQQ = lm.permutation_matrix_square(self.Q, self.Q)
        self.q2s = lm.mpoly_square_expos(self.qts)
        self.Ct = lm.mpoly_def_int_coef_L(self.q2s, 0, self.a, self.b) @ self.RQQ
        self.q2 = lm.mpoly_def_int_expos(self.q2s, 0)[0]
        self.K = lm.commutation_matrix(len(self.q2s[0]), len(self.q2s[1]))
        self.Kt = np.identity(len(self.q2) ** 2) + self.K
        self.A = self.Ct @ np.kron(self.Lt, self.Lt)
        self.B = self.Ct @ self.Kt @ np.kron(self.Lt, self.Bt)
        self.C = self.Ct @ np.kron(self.Bt, self.Bt)

        # delay interval
        self.s = np.arange(-100, 100, 1)

    def integrate_delay(self, alpha_k, beta_k, k_left, k_right):
        # re-adjusting the coefficients to be over the same time-axis
        trans_matrix_left = np.asarray(
            [
                [1, -k_left, np.power(k_left, 2), -np.power(k_left, 3)],
                [0, 1, -2 * k_left, 3 * np.power(k_left, 2)],
                [0, 0, 1, -3 * k_left],
                [0, 0, 0, 1],
            ]
        )

        trans_matrix_right = np.asarray(
            [
                [1, -k_right, np.power(k_right, 2), -np.power(k_right, 3)],
                [0, 1, -2 * k_right, 3 * np.power(k_right, 2)],
                [0, 0, 1, -3 * k_right],
                [0, 0, 0, 1],
            ]
        )

        alpha_k = np.squeeze(trans_matrix_left @ np.reshape(alpha_k, (4, 1)), axis=1)
        beta_k = np.squeeze(trans_matrix_right @ np.reshape(beta_k, (4, 1)), axis=1)

        # observation
        observation_coef = (
            self.A @ np.kron(alpha_k, alpha_k)
            - self.B @ np.kron(alpha_k, beta_k)
            + self.C @ np.kron(beta_k, beta_k)
        )

        # squared Error
        J = lm.Poly(observation_coef, self.q2)
        js = J.eval(self.s)

        # computing the delay
        delay_id = np.argmin(js)

        return self.s[delay_id]

    def estimate_delay(self):
        """
        Method that  estimates the delay between two sound signals based on the local polynomial approximation of their
        corresponding LCRs. The method uses the class attributes and saves a csv file with the unique delays.
        """
        azimuth = self.azimuth
        frequencies = self.frequencies

        # reading the coefficient files
        coeff_left = []
        coeff_right = []

        for i, freq in enumerate(frequencies):
            coeff_left.append(
                np.genfromtxt(
                    self.coefficients_saving_path
                    + "/"
                    + self.signal_id
                    + "_coeff_left_Az_"
                    + str(azimuth)
                    + "_freq_"
                    + str(freq)
                    + ".csv",
                    delimiter=",",
                    skip_header=False,
                )
            )
            coeff_right.append(
                np.genfromtxt(
                    self.coefficients_saving_path
                    + "/"
                    + self.signal_id
                    + "_coeff_right_Az_"
                    + str(azimuth)
                    + "_freq_"
                    + str(freq)
                    + ".csv",
                    delimiter=",",
                    skip_header=False,
                )
            )
        # stacking coefficients from different frequencies in the third dimension
        coeff_left = np.stack(coeff_left, axis=2)
        coeff_right = np.stack(coeff_right, axis=2)

        # window length parameter
        window_half_len = 25

        # array that stores the estimated delays
        total_delay = np.zeros(coeff_left.shape[0])
        delay_per_freq = np.zeros((len(frequencies), coeff_left.shape[0]))
        overall_conf = np.zeros((len(frequencies), coeff_left.shape[0]))
        max_overall_conf = 0

        for k in range(window_half_len, coeff_left.shape[0] - window_half_len):
            # retrieving the coefficients inside our current window
            coeff_left_window = coeff_left[
                k - window_half_len : k + window_half_len, :, :
            ]
            coeff_right_window = coeff_right[
                k - window_half_len : k + window_half_len, :, :
            ]

            coeff_left_list = np.split(coeff_left_window, len(frequencies), axis=2)
            coeff_right_list = np.split(coeff_right_window, len(frequencies), axis=2)

            # computing all the distances between polynomials within the window
            for i in range(len(coeff_left_list)):
                curr_coeff_left = np.squeeze(coeff_left_list[i], axis=2)
                curr_coeff_right = np.squeeze(coeff_right_list[i], axis=2)
                distances = distance.cdist(
                    curr_coeff_left, curr_coeff_right, "euclidean"
                )

                # finding the two most similar polynomials inside the window
                id_min_poly_left = np.where(distances == distances.min())[0][0]
                id_min_poly_right = np.where(distances == distances.min())[1][0]

                # retrieving the corresponding coefficients
                coeff_min_left = curr_coeff_left[id_min_poly_left, :]
                coeff_min_right = curr_coeff_right[id_min_poly_right, :]

                # checking some conditions
                condition_1 = coeff_min_left[0] > 0 and coeff_min_right[0] > 0
                condition_2 = coeff_min_left[1] > 1.2e-6 and coeff_min_right[1] > 1.2e-6
                condition_3 = (
                    2 * coeff_min_left[2] < 1e-12 and 2 * coeff_min_right[2] < 1e-12
                )

                if condition_1 and condition_2 and condition_3:
                    # computing the confidence
                    overall_conf[i, k] = 1 / distances.min()

                    # first version of the delay: just the difference between the indexes where the minimums occur
                    # delay_per_freq[i, k] = abs(np.where(distances == distances.min())[0][0] - np.where(distances == distances.min())[1][0])

                    # current version of the delay: integration method
                    delay_per_freq[i, k] = abs(
                        self.integrate_delay(
                            coeff_min_left,
                            coeff_min_right,
                            id_min_poly_left,
                            id_min_poly_right,
                        )
                    )
                else:
                    delay_per_freq[i, k] = delay_per_freq[i, k - 1]

            # decide upon a final delay for the whole signal
            id_max_conf = np.where(overall_conf[:, k] == overall_conf[:, k].max())[0][0]
            if overall_conf[:, k].max() > max_overall_conf:
                total_delay[k] = delay_per_freq[id_max_conf, k]
                max_overall_conf = overall_conf[:, k].max()
            else:
                total_delay[k] = total_delay[k - 1]

        # saving confidences and delay estimates
        np.save(
            self.delay_saving_path
            + "/"
            + self.signal_id
            + "_all_delays_Az_"
            + str(azimuth)
            + ".npy",
            total_delay,
        )
        np.save(
            self.delay_saving_path
            + "/"
            + self.signal_id
            + "_delays_per_freq_Az_"
            + str(azimuth)
            + ".npy",
            delay_per_freq,
        )
        np.save(
            self.delay_saving_path
            + "/"
            + self.signal_id
            + "_over_conf_per_freq_Az_"
            + str(azimuth)
            + ".npy",
            overall_conf,
        )


# if __name__ == '__main__':
#
# # defining the simulation parameters
# # onset model considered. Either 'decaying_sinusoid' or 'gammatone'
# onset_model = 'decaying_sinusoid'
# # window model considered. Either 'gamma' or 'exponential'
# window_model = 'gamma'
# # number of frequencies used simultaneously (linear combination)
# n_freq = 1
# # list of frequencies used simultaneously
# filter_bank_frequencies = [[80.0], [120.0], [160.0], [200.0], [240.0]]  # list of lists
# # list of onset decays. Follows the same order as the list of frequencies
# onset_decays = [0.99]
# # window decay used
# window_decay = 0.999
# # list of azimuths used
# azimuth = [65, 80]
#
# all_signals = ['ists_sig_1', 'ists_sig_2', 'ists_sig_3', 'ists_sig_4', 'ists_sig_5', 'ists_sig_6', 'ists_sig_7',
#                'ists_sig_8', 'ists_sig_9', 'ists_sig_10', 'ists_sig_11', 'ists_sig_12', 'ists_sig_13', 'ists_sig_14',
#                'ists_sig_15', 'ists_sig_16', 'ists_sig_17', 'ists_sig_18', 'ists_sig_19', 'ists_sig_20',
#                'ists_sig_21', 'ists_sig_22']
#
# for az in azimuth:
#     # data path
#     azimuth_data_path = 'Dataset from ISTS/Az_' + str(az)
#
#     # list of speech signals in the data path
#
#
#     # creating the sub-directories that contains the results (LCRs, polynomial coefficients and delay estimates)
#     pathlib.Path(azimuth_data_path + '/Prelim delays').mkdir(parents=True, exist_ok=True)
#     # lcr_saving_path = azimuth_data_path + '/LCR'
#     coeff_saving_path = azimuth_data_path + '/Coefficients'
#     delay_saving_path = azimuth_data_path + '/Prelim delays'
#     #
#     # # keeping track of the sample and frequency used for the delay estimation procedure
#     # delay_info = {}
#     current_del = []
#     for signal in all_signals:
#         signal_id = signal
#
#         # delays = np.load(delay_saving_path + '/' + signal_id + '_all_delays_Az_' + str(az) + '.npy')
#         # plt.scatter(az, np.median(delays))
#         # estimating the delay from the polynomial fit
#         delay = DelayEstimationSigStroke(frequencies=filter_bank_frequencies, azimuth=az,
#                                          delay_saving_path=delay_saving_path,
#                                          coefficients_saving_path=coeff_saving_path, signal_id=signal_id)
#         delay.estimate_delay()
