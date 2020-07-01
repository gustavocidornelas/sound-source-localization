"""
Script that creates a training dataset from segments of the ISTS signal to validate the algorithm

----------
Author: Gustavo Cid Ornelas, ETH Zurich, May 2020
"""

import numpy as np
import pathlib
import scipy.signal
import os
import matplotlib.pyplot as plt

from scipy.io import wavfile


def convolve_hrir(azimuth, speech_signal):
    """
    Function that convolves the single-channel speech signal with the HRIR of the corresponding azimuth and returns the
    multi-channel audio signal corresponding to the right and left
    """
    # getting the correct HRIR signal
    hrir = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/HRIR/@44.1kHz/HRIR_Az_"
        + str(azimuth)
        + ".csv",
        delimiter=",",
    )
    hrir_left = hrir[:, 0]
    hrir_right = hrir[:, 1]

    # convolving the impulse responses with the speech signal
    speech_left = np.convolve(hrir_left, speech_signal, mode="same")
    speech_right = np.convolve(hrir_right, speech_signal, mode="same")

    # multi-channel signal
    return np.vstack((speech_left, speech_right))


def pad_speech(speech_signal):
    """
    Function that pads a random vector (white Gaussian noise) to the beginning of the speech signal
    """
    # computing the noise variance at the beginning of the speech signal
    noise_var = np.var(speech_signal[:500])

    pad = np.random.normal(
        0, np.sqrt(noise_var), 30000
    )  # same length as the burn-in period for the LCR (these samples are thrown away later)

    padded_speech_signal = np.hstack((pad, speech_signal))

    return padded_speech_signal


if __name__ == "__main__":
    # fixing the random number generator seed
    np.random.seed(33)

    # creating the directory that contains the data
    pathlib.Path("Dataset from ISTS").mkdir(parents=True, exist_ok=True)

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
        # creating the directory for the current azimuth
        pathlib.Path("Dataset from ISTS/Az_" + str(az)).mkdir(
            parents=True, exist_ok=True
        )

        # convolving signals with HRIR
        for i in range(1, 23):
            speech_signal = np.load(
                "/Users/gustavocidornelas/Desktop/sound-source/utils/ists_s"
                + str(i)
                + ".npy"
            )

            # padding the beginning of the signal with noise
            speech_signal = pad_speech(speech_signal)

            # convolving single channel signal with HRIR to obtain multi-channel signal
            multi_channel_speech = np.transpose(
                convolve_hrir(azimuth=az, speech_signal=speech_signal)
            )

            np.save(
                "Dataset from ISTS/Az_"
                + str(az)
                + "/ists_sig_"
                + str(i)
                + "_Az_"
                + str(az)
                + ".npy",
                multi_channel_speech,
            )
