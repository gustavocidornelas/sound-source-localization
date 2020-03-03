"""
Estimation of the ITD from the sound onsets with a simple delay threshold. A relative threshold value is used.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""
import numpy as np

if __name__ == '__main__':

    # azimuth list
    azimuth = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65,
               80]

    # relative to the maximum signal amplitude in dB
    rel_thresh_dB = -38

    all_delays = []

    for az in azimuth:
        # reading the audio file
        audio = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_' +
                              str(az) + '.csv',
                              delimiter=',')
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

        delay = (onset_times_L - onset_times_R)/44.1
        print('Delay: ' + str(delay) + ' msec')
        all_delays.append(delay)

    all_delays = np.array(all_delays)
    np.save('delays_threshold.npy', all_delays)
