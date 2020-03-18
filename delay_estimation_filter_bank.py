"""
----------
Author: Gustavo Cid Ornelas, ETH Zurich, November 2019

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


class DelayEstimationFilterBank:
    """
    Class that contains the method to perform the delay estimation in the case where a filter bank is being used.

    Parameters:
    ----------
    frequencies (list): frequencies of the onset models used to generate the LCRs that will be fit with the polynomial
    azimuth (float): float that represents the value of the azimuth of the sound source (used only to save the file with
                  the correct name)
    """
    def __init__(self, frequencies, azimuth):
        self.frequencies = frequencies
        self.azimuth = azimuth

    def estimate_delay(self):
        """
        Method that  estimates the delay between two sound signals based on the local polynomial approximation of their
        corresponding LCRs. The method uses the class attributes and saves a csv file with the unique delays.
        """
        azimuth = self.azimuth
        frequencies = self.frequencies

        print('Estimating the delays...')

        # reading the coefficient files
        coeff_left = []
        coeff_right = []
        for i, freq in enumerate(frequencies):
            coeff_left.append(np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/coeff_left_Az_' +
                                            str(azimuth) + '_freq_' + str(freq) + '.csv',
                                            delimiter=',', skip_header=False))
            coeff_right.append(np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/coeff_right_Az_' +
                                             str(azimuth) + '_freq_' + str(freq) + '.csv',
                                             delimiter=',', skip_header=False))
        # stacking coefficients from different frequencies in the third dimension
        coeff_left = np.stack(coeff_left, axis=2)
        coeff_right = np.stack(coeff_right, axis=2)

        # some useful computations and parameters
        # number of samples in sequence used to do the continuous delay estimation
        N = 50
        # window size for the 2nd degree polynomial approximation
        a0 = -40.0
        b0 = 40.0
        A = np.array([[1, 0, 0],
                      [1, a0, np.power(a0, 2)],
                      [1, b0, np.power(b0, 2)]])
        A_inv = np.linalg.inv(A)

        # array that stores the estimated delays
        total_delay = np.zeros(coeff_left.shape[0])
        delay = 0
        count = 0
        continuous_delay_status = 0  # used to check if it should perform continuous delay estimation or not

        # main loop, going through each sample coefficients
        for k in range(coeff_left.shape[0]):
            # retrieving the current sample's coefficients
            coeff_left_k = coeff_left[k, :, :]
            coeff_right_k = coeff_right[k, :, :]

            # checking the conditions for each LCR polynomial
            first_condition = np.logical_and(coeff_left_k[1, :] > 1.3e-6, coeff_right_k[1, :] > 1.3e-6)
            second_condition = np.logical_and(coeff_left_k[2, :] < 0.2e-13, coeff_right_k[2, :] > 0.2e-13)

            decision = np.logical_and(first_condition, second_condition)

            # if the condition is satisfied for any frequency in the filter bank
            if any(decision) or continuous_delay_status == 1:
                # frequency to be used
                if count == 0:
                    freq_id = np.argmax(decision)
                #print('Frequency: ' + str(frequencies[freq_id]))

                #print(k)
                # approximating the 3rd degree polynomials by 2nd degree polynomials (coefficients beta) on a smaller window
                beta_left = np.dot(A_inv, np.array([[coeff_left_k[0, freq_id]],
                                                    [coeff_left_k[0, freq_id] + coeff_left_k[1, freq_id] * a0 +
                                                     coeff_left_k[2, freq_id] * np.power(a0, 2) + coeff_left_k[3,
                                                                                                               freq_id]
                                                     * np.power(a0, 3)],
                                                    [coeff_left_k[0, freq_id] + coeff_left_k[1, freq_id] * b0 +
                                                     coeff_left_k[2, freq_id] * np.power(b0, 2) + coeff_left_k[3,
                                                                                                               freq_id]
                                                     * np.power(b0, 3)]
                                                    ]))
                beta_right = np.dot(A_inv, np.array([[coeff_right_k[0, freq_id]],
                                                     [coeff_right_k[0, freq_id] + coeff_right_k[1, freq_id] * a0 +
                                                      coeff_right_k[2, freq_id] * np.power(a0, 2) + coeff_right_k[3,
                                                                                                                  freq_id]
                                                      * np.power(a0, 3)],
                                                     [coeff_right_k[0, freq_id] + coeff_right_k[1, freq_id] * b0 +
                                                      coeff_right_k[2, freq_id] * np.power(b0, 2) + coeff_right_k[3,
                                                                                                                  freq_id]
                                                      * np.power(b0, 3)]
                                                     ]))

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
                        # count = 0
                    # print(delay)

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
                    delay = np.mean(total_delay[k - N:k])
                    total_delay[k - N:k] = np.median(total_delay[k - N:k])

            total_delay[k] = delay

        np.save('celoko_teste_all_delays_Az_' + str(azimuth) + '.npy', total_delay)
        # print(np.unique(total_delay)/44.1)
        print(np.median(np.unique(total_delay) / 44.1))
        np.savetxt('unique_delays_Az_' + str(azimuth) + '_freq_' + str(frequencies) + '.csv', np.unique(total_delay),
                   delimiter=',')
        # plt.figure()
        # plt.plot(np.asarray(range(total_delay.shape[0]))/44100, total_delay/44.1, linewidth=1.8)
        # plt.ylabel('ITD [ms]')
        # plt.xlabel('Time [s]')
        # plt.grid(ls='--', c='.5')

        # plt.show()


if __name__ == '__main__':
    p = DelayEstimationFilterBank(frequencies=[[80.0], [115.0], [150.0], [185.0]], azimuth=-45)
    p.estimate_delay()

    LCR_80 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_[0.99]_gamma_0.999_Az_-45_'
                           'freq_[80.0].csv', delimiter=',')
    LCR_80 = LCR_80[30000:, :]
    LCR_140 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_[0.99]_gamma_0.999_Az_-45_'
                           'freq_[115.0].csv', delimiter=',')
    LCR_140 = LCR_140[30000:, :]
    LCR_200 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_[0.99]_gamma_0.999_Az_-45_'
                           'freq_[150.0].csv', delimiter=',')
    LCR_200 = LCR_200[30000:, :]
    LCR_260 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_[0.99]_gamma_0.999_Az_-45_'
                           'freq_[185.0].csv', delimiter=',')
    LCR_260 = LCR_260[30000:, :]

    total_delays = np.load('teste_all_delays_Az_-45.npy')

    fig, axs = plt.subplots(5)
    axs[0].plot(np.asarray(range(LCR_80.shape[0])), LCR_80[:, 1])
    axs[0].plot(np.asarray(range(LCR_80.shape[0])), LCR_80[:, 2])
    axs[1].plot(np.asarray(range(LCR_80.shape[0])), LCR_140[:, 1])
    axs[1].plot(np.asarray(range(LCR_80.shape[0])), LCR_140[:, 2])
    axs[2].plot(np.asarray(range(LCR_80.shape[0])), LCR_200[:, 1])
    axs[2].plot(np.asarray(range(LCR_80.shape[0])), LCR_200[:, 2])
    axs[3].plot(np.asarray(range(LCR_80.shape[0])), LCR_260[:, 1])
    axs[3].plot(np.asarray(range(LCR_80.shape[0])), LCR_260[:, 2])
    axs[4].plot(np.asarray(range(total_delays.shape[0])), total_delays / 44.1)

    # median filtering in the transitions
    print('carai borracha: ' + str(np.median(np.unique(total_delays[208411:209029])) / 44.1))
    print('carai borracha 2: ' + str(np.median(np.unique(total_delays[235790:235820])) / 44.1))

    axs[0].set_title('80 Hz')
    axs[1].set_title('140 Hz')
    axs[2].set_title('200 Hz')
    axs[3].set_title('260 Hz')
    plt.show()



