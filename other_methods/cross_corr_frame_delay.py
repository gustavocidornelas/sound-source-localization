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


def convert_to_compare(signal_id, frame_size, az, fs=44.1):
    signal_id = signal_id.split("_Az", 1)[0]
    our_delays_path = (
        "/Users/gustavocidornelas/Desktop/Delays - confidence integration/Az_" + str(az)
    )
    try:
        our_delays = np.load(
            our_delays_path + "/" + signal_id + "_all_delays_Az_" + str(az) + ".npy"
        )
        our_confidences = np.load(
            our_delays_path
            + "/"
            + signal_id
            + "_over_conf_per_freq_Az_"
            + str(az)
            + ".npy"
        )
    except:
        print(az)
        return np.zeros(1), 0

    L = our_delays.shape[0]
    num_zeros = L - int(L / frame_size) * frame_size + 1

    our_delays = our_delays[:-num_zeros]
    our_confidences = our_confidences[:, :-num_zeros]

    plt.plot(range(our_delays.shape[0]), our_delays / 44.1, "r", label="Our method")

    # getting the id of our first estimate
    try:
        index = np.where(our_confidences != 0)[1].min()

        our_points = our_delays[index::frame_size] / fs
    except:
        our_points = np.zeros(1)
        index = 0

    return our_points, index


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
    audio_L = audio[30000:, 0]
    audio_R = audio[30000:, 1]
    L = audio_L.shape[0]

    fs = fs * 1000
    delay_sum = 0.0

    # zero padding the signals at the end
    num_zeros = L - int(L / frame_size) * frame_size
    audio_L = np.hstack((audio_L, np.zeros(num_zeros)))
    audio_R = np.hstack((audio_R, np.zeros(num_zeros)))

    delay_per_frame = np.zeros(int(L / frame_size))

    # going through the signal frame by frame
    for i in range(int(L / frame_size)):
        # getting the frames
        frame_L = audio_L[i * frame_size : (i + 1) * frame_size]
        frame_R = audio_R[i * frame_size : (i + 1) * frame_size]

        # calculating the cross-correlation between the two frames

        corr = signal.correlate(frame_L, frame_R, mode="same") / np.sqrt(
            signal.correlate(frame_R, frame_R, mode="same")[int(frame_size / 2)]
            * signal.correlate(frame_L, frame_L, mode="same")[int(frame_size / 2)]
        )
        delay_arr = np.linspace(
            -0.5 * frame_size / fs, 0.5 * frame_size / fs, frame_size
        )
        delay_sum = delay_sum + delay_arr[np.argmax(corr)]
        delay_per_frame[i] = delay_arr[np.argmax(corr)]

    delay = delay_sum / int(L / frame_size)

    delays_comp = []
    for i in range(delay_per_frame.shape[0]):
        delays_comp.extend(np.repeat(delay_per_frame[i], frame_size))

    data_path = (
        "/Users/gustavocidornelas/Desktop/Delays - confidence integration/Az_-80"
    )
    file_path = data_path + "/ists_sig_14_all_delays_Az_-80.npy"
    delays_ours = np.load(file_path)
    delays_ours = delays_ours[: -num_zeros + 1]

    # plt.figure()
    plt.plot(
        range(len(delays_comp)),
        abs(np.asarray(delays_comp)) * 1e3,
        label="Cross-correlation",
    )
    # plt.plot(range(delays_ours.shape[0]), delays_ours / 44.1, 'r', label='Our method')
    # #plt.plot(range(delays_ours.shape[0]), np.repeat(0.436, delays_ours.shape[0]),'--', label='Theoretical')
    plt.legend()
    plt.show()

    # return abs(delay) * 1e3
    return (
        abs(delay_per_frame) * 1e3
    )  # delay_per_frame[(delay_per_frame > np.mean(delay_per_frame) - np.std(delay_per_frame)) & (delay_per_frame < np.mean(delay_per_frame) + np.std(delay_per_frame))] * 1e3


if __name__ == "__main__":
    # azimuth list
    azimuth = [
        -80,
        -65,
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
        65,
        80,
    ]
    frame = 100

    for az in azimuth:
        print("Azimuth " + str(az))
        # path to the data
        azimuth_data_path = (
            "/Users/gustavocidornelas/Desktop/sound-source/mini-world/Dataset from ISTS/Az_"
            + str(az)
        )

        # list of speech signals in the data path
        all_signals = [
            file for file in os.listdir(azimuth_data_path) if file.endswith(".npy")
        ]

        # creating the directory to save the delays
        pathlib.Path(
            azimuth_data_path + "/Delays cross-correlation per frame method"
        ).mkdir(parents=True, exist_ok=True)
        pathlib.Path(azimuth_data_path + "/Delays comp. our method").mkdir(
            parents=True, exist_ok=True
        )

        # array that stores all delays for the current azimuth (one per signal)
        # all_delays = np.zeros(len(all_signals))
        total_delays = []
        our_method_delays = []

        for i, signal_id in enumerate(all_signals):
            audio_signal = np.load(azimuth_data_path + "/" + signal_id)
            # all_delays[i] = estimate_delay_cross_corr_frame(audio_signal, frame_size=100)
            cc_delays = estimate_delay_cross_corr_frame(audio_signal, frame_size=frame)
            our_delays, index = convert_to_compare(
                signal_id=signal_id, frame_size=frame, az=az
            )
            len_delays = our_delays.shape[0]
            total_delays.extend(cc_delays[cc_delays.shape[0] - len_delays :])
            our_method_delays.extend(our_delays)
            # total_delays.extend(cc_delays)
        our_method_delays = np.asarray(our_method_delays)
        our_method_delays = our_method_delays[
            np.where(our_method_delays < our_method_delays.max())
        ]
        print(0)
        # plt.scatter(np.repeat(az, len(our_method_delays)) + 0.05, our_method_delays, label='our')
        # plt.scatter(np.repeat(az, len(total_delays)), total_delays, label='cc')
        # plt.legend()
        # plt.show()

        # save results
        # np.save(azimuth_data_path + '/Delays cross-correlation per frame method/ists_cross_corr_frame_delays_Az_' + str(az) + '.npy',
        #         all_delays)
        np.save(
            azimuth_data_path
            + "/Delays cross-correlation per frame method/ists_cross_corr_frame_delays_Az_"
            + str(az)
            + ".npy",
            total_delays,
        )
        np.save(
            azimuth_data_path
            + "/Delays comp. our method/ists_our_delays_Az_"
            + str(az)
            + ".npy",
            our_method_delays,
        )
