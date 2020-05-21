"""
Estimation of the ITD from the cross-correlation between frames (instead of the whole signal, done
 in cross_corr_delay.py).

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

from scipy import signal


def estimate_delay_cross_corr_frame(audio, fs=44.1, frame_size=44100):
    """
    Function that estimates the delay by computing the cross-correlation between the two audio signals, frame by frame

    Parameters:
    ----------
    audio (array): stereo audio signal
    fs (float): sampling frequency in kHz. Default is 44.1 kHz
    frame_size (int): frame size. Default is 44100 (1 sec)

    Returns:
    ----------
    delay (float): delay estimate in msec
    """

    # separating the signals from the left and from the right
    audio_L = audio[:, 0]
    audio_R = audio[:, 1]
    L = audio_L.shape[0]

    fs = fs * 1000
    delay_sum = 0.0

    # zero padding the signals at the end
    num_zeros = L - int(L / frame_size) * frame_size
    audio_L = np.hstack((audio_L, np.zeros(num_zeros)))
    audio_R = np.hstack((audio_R, np.zeros(num_zeros)))

    # going through the signal frame by frame
    for i in range(int(L / frame_size)):
        # getting the frames
        frame_L = audio_L[i * frame_size: (i + 1) * frame_size]
        frame_R = audio_R[i * frame_size: (i + 1) * frame_size]

        # calculating the cross-correlation between the two frames

        corr = signal.correlate(frame_L, frame_R, mode='same') / np.sqrt(signal.correlate(frame_R, frame_R,
                                                                                          mode='same')[
                                                                             int(frame_size / 2)] *
                                                                         signal.correlate(frame_L, frame_L,
                                                                                          mode='same')[
                                                                             int(frame_size / 2)])
        delay_arr = np.linspace(-0.5 * frame_size / fs, 0.5 * frame_size / fs, frame_size)
        delay_sum = delay_sum + delay_arr[np.argmax(corr)]

    delay = delay_sum / int(L / frame_size)

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
        pathlib.Path(azimuth_data_path + '/Delays cross-correlation per frame method').mkdir(parents=True, exist_ok=True)

        # array that stores all delays for the current azimuth (one per signal)
        all_delays = np.zeros(len(all_signals))

        for i, signal_id in enumerate(all_signals):
            audio_signal = np.load(azimuth_data_path + '/' + signal_id)
            all_delays[i] = estimate_delay_cross_corr_frame(audio_signal, frame_size=100)

        # save results
        np.save(azimuth_data_path + '/Delays cross-correlation per frame method/ists_cross_corr_frame_delays_Az_' + str(az) + '.npy',
                all_delays)
