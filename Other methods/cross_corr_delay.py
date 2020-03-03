"""
Estimation of the ITD from the cross-correlation.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

if __name__ == '__main__':
    # azimuth list
    azimuth = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65,
               80]

    all_delays = []
    for az in azimuth:
        # reading the audio file
        audio = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/HRIR/HRIR_Az_' +
                              str(az) + '.csv',
                              delimiter=',')
        # separating the signals from the left and from the right
        audio_L = audio[:, 0]
        audio_R = audio[:, 1]
        L = audio_L.shape[0]

        # calculating the cross-correlation between the two signals
        fs = 44100.0
        corr = signal.correlate(audio_L, audio_R, mode='same') / np.sqrt(signal.correlate(audio_R, audio_R,
                                                                                          mode='same')[int(L/2)] *
                                                                         signal.correlate(audio_L, audio_L,
                                                                                          mode='same')[int(L/2)])
        delay_arr = np.linspace(-0.5 * L / fs, 0.5 * L / fs, L)
        delay = delay_arr[np.argmax(corr)]
        print(delay * 1e3)
        all_delays.append(delay * 1e3)

    all_delays = np.array(all_delays)
    np.save('delays_cross_corr.npy', all_delays)

    #plt.figure()
    #plt.plot(range(corr.shape[0]), corr)
    #plt.show()
