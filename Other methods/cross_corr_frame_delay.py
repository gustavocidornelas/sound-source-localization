"""
Estimation of the ITD from the cross-correlation between frames (instead of the whole signal, done
 in cross_corr_delay.py).

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

if __name__ == '__main__':
    # azimuth list
    azimuth = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65,
               80]
    # sampling frequency
    fs = 44100.0

    # frame size (in number of samples)
    frame_size = 44100

    all_delays = []
    for az in azimuth:
        # reading the audio file
        audio = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_' +
                              str(az) + '.csv',
                              delimiter=',')
        # separating the signals from the left and from the right
        audio_L = audio[:, 0]
        audio_R = audio[:, 1]
        L = audio_L.shape[0]

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
                                                                                          mode='same')[int(frame_size/2)] *
                                                                            signal.correlate(frame_L, frame_L,
                                                                                             mode='same')[int(frame_size/2)])
            delay_arr = np.linspace(-0.5 * frame_size / fs, 0.5 * frame_size / fs, frame_size)
            delay = delay_arr[np.argmax(corr)]
            print(delay * 1e3)
            all_delays.append(delay * 1e3)

    #all_delays = np.array(all_delays)
    #np.save('delays_cross_corr.npy', all_delays)

    #plt.figure()
    #plt.plot(range(corr.shape[0]), corr)
    #plt.show()
