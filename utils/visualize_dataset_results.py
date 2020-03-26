"""
Script to visualize the speech signal, corresponding LCRs and delay estimates for the signals on the speech dataset
(built over the TIMIT dataset). Displays a single figure with multiple subplots.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    # sampling rate
    fs = 16000.0
    # burn-in period
    burn_in = 0

    # visualize results for the following azimuth and the following speech signal
    azimuth = -80
    signal_id = 'f_sig_2'

    # setting up the paths
    azimuth_data_path = '/Users/gustavocidornelas/Desktop/sound-source/Dataset/Az_' + str(azimuth)
    lcr_path = azimuth_data_path + '/LCR'
    coeff_path = azimuth_data_path + '/Coefficients'
    delay_path = azimuth_data_path + '/Delays'

    # retrieving results for the specified signal
    audio_signal = np.load(azimuth_data_path + '/' + signal_id + '_Az_' + str(azimuth) + '.npy')
    audio_signal = audio_signal[burn_in:, :]
    all_lcr_files = [file for file in os.listdir(lcr_path) if file.startswith(signal_id)]
    all_delay_files = [file for file in os.listdir(delay_path) if file.startswith(signal_id)]

    # creating the plot
    fig, axs = plt.subplots(2 + len(all_lcr_files))  # one for the audio signal, one for the delays and all the LCRs
    axs[0].set_title('Audio signal')
    axs[0].plot(np.asarray(range(audio_signal.shape[0])) / fs, audio_signal[:, 0], 'b')
    axs[0].plot(np.asarray(range(audio_signal.shape[0])) / fs, audio_signal[:, 1], 'r')

    # plotting the LCRs
    for i, lcr_file in enumerate(all_lcr_files):
        lcr = np.genfromtxt(lcr_path + '/' + lcr_file, delimiter=',', skip_header=True)
        lcr = lcr[burn_in:, :]

        # onset frequency to generate this LCR
        freq = lcr_file.split('_freq_', 1)[1][:-4]

        # plotting
        axs[i + 1].plot(np.asarray(range(lcr.shape[0])) / fs, lcr[:, 1], 'b')
        axs[i + 1].plot(np.asarray(range(lcr.shape[0])) / fs, lcr[:, 2], 'r')
        axs[i + 1].set_title(freq + ' Hz')

    # plotting the delays
    delays = np.load(delay_path + '/' + all_delay_files[0])
    axs[-1].plot(np.asarray(range(delays.shape[0])) / fs, delays, 'k')
    axs[-1].set_xlabel('Time [s]')

    plt.show()
