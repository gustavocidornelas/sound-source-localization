"""
Estimation of the ITD from the sound onsets with a simple delay threshold. A relative threshold value is used.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""

import numpy as np
import os
import pathlib


def estimate_delay_threshold(audio, fs=44.1, rel_thresh_dB=-30):
    """
    Function that estimates the delay via the threshold method

    Parameters:
    ----------
    audio (array): stereo audio signal
    fs (float): sampling frequency in kHz. Default is 44.1 kHz
    rel_thresh_dB (float): how many dB below peak of audio signal is the threshold. Default is -30 dB

    Returns:
    ----------
    delay (float): delay estimate in msec
    """
    # separating the signals from the left and from the right
    audio_L = audio[:, 0]
    audio_R = audio[:, 1]

    # defining the threshold value
    threshold_dB = 20 * np.log10(np.max(audio_L)) + rel_thresh_dB
    threshold = np.power(10, threshold_dB / 20)

    # aux variables to check if it is the first time surpassing the thresholds for each signal
    status_L = 0
    status_R = 0

    # going through the audio signal
    for k in range(audio.shape[0]):
        current_L = audio_L[k]
        current_R = audio_R[k]

        if current_L >= threshold and status_L == 0:
            onset_times_L = k
            status_L = 1
        if current_R >= threshold and status_R == 0:
            onset_times_R = k
            status_R = 1

    delay = (onset_times_L - onset_times_R) / fs

    return abs(delay)


if __name__ == "__main__":

    # azimuth list
    azimuths = [
        -80,
        -65,
        -55,
        -45,
        -40,
        -35,
        -30,
        -25,
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        55,
        65,
        80,
    ]

    for az in azimuths:
        print("Azimuth " + str(az))
        # path to the data
        azimuth_data_path = (
            "/Users/gustavocidornelas/Desktop/sound-source/data/new Dataset from ISTS/Az_"
            + str(az)
        )

        # list of speech signals in the data path
        all_signals = [
            file for file in os.listdir(azimuth_data_path) if file.endswith(".npy")
        ]

        # creating the directory to save the delays
        pathlib.Path(azimuth_data_path + "/Delays threshold method").mkdir(
            parents=True, exist_ok=True
        )

        # array that stores all delays for the current azimuth (one per signal)
        all_delays = np.zeros(len(all_signals))

        for i, signal in enumerate(all_signals):
            audio_signal = np.load(azimuth_data_path + "/" + signal)
            all_delays[i] = estimate_delay_threshold(audio_signal, rel_thresh_dB=-20)

        # save results
        np.save(
            azimuth_data_path
            + "/Delays threshold method/ists_threshold_delays_Az_"
            + str(az)
            + ".npy",
            all_delays,
        )
