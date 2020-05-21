"""
Estimation of the ITD from the cross-correlation.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from scipy import signal


def estimate_delay_cross_correlation(audio, fs=44.1):
    """
    Function that estimates the delay by computing the cross-correlation between the two audio signals

    Parameters:
    ----------
    audio (array): stereo audio signal
    fs (float): sampling frequency in kHz. Default is 44.1 kHz

    Returns:
    ----------
    delay (float): delay estimate in msec
    """
    # separating the signals from the left and from the right
    audio_L = audio[:, 0]
    audio_R = audio[:, 1]
    L = audio_L.shape[0]

    # calculating the cross-correlation between the two signals
    fs = fs * 1000
    corr = signal.correlate(audio_L, audio_R, mode='same') / np.sqrt(signal.correlate(audio_R, audio_R, mode='same')[int(L / 2)]
                                                                     * signal.correlate(audio_L, audio_L, mode='same')[int(L / 2)])
    delay_arr = np.linspace(-0.5 * L / fs, 0.5 * L / fs, L)
    delay = delay_arr[np.argmax(corr)]

    return abs(delay) * 1e3


if __name__ == '__main__':
    # azimuth list
    azimuth = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65,
               80]

    for az in azimuth:
        print('Azimuth ' + str(az))
        # path to the data
        azimuth_data_path = '/Users/gustavocidornelas/Desktop/sound-source/mini-world/Dataset from ISTS/Az_' + str(az)

        # list of speech signals in the data path
        all_signals = [file for file in os.listdir(azimuth_data_path) if file.endswith('.npy')]

        # creating the directory to save the delays
        pathlib.Path(azimuth_data_path + '/Delays cross-correlation method').mkdir(parents=True, exist_ok=True)

        # array that stores all delays for the current azimuth (one per signal)
        all_delays = np.zeros(len(all_signals))

        for i, signal_id in enumerate(all_signals):
            audio_signal = np.load(azimuth_data_path + '/' + signal_id)
            all_delays[i] = estimate_delay_cross_correlation(audio_signal)

        # save results
        np.save(azimuth_data_path + '/Delays cross-correlation method/ists_cross_corr_delays_Az_' + str(az) + '.npy', all_delays)
