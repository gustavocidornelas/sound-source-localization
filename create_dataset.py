"""
Script that creates a training dataset from the TIMIT corpus to validate the algorithm

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

from scipy.io import wavfile


def convolve_hrir(azimuth, speech_signal):
    """
    Function that convolves the single-channel speech signal with the HRIR of the corresponding azimuth and returns the
    multi-channel audio signal corresponding to the right and left
    """
    # getting the correct HRIR signal
    hrir = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/HRIR/HRIR_Az_' + str(azimuth) +
                         '.csv', delimiter=',')
    hrir_left = hrir[:, 0]
    hrir_right = hrir[:, 1]

    # convolving the impulse responses with the speech signal
    speech_left = np.convolve(hrir_left, speech_signal, mode='same')
    speech_right = np.convolve(hrir_right, speech_signal, mode='same')

    # multi-channel signal
    return np.vstack((speech_left, speech_right))


if __name__ == '__main__':
    # fixing the random number generator seed
    np.random.seed(33)

    # dictionary with the information of the files chosen per azimuth
    az_info = {'DR1': 0, 'DR2': 0, 'DR3': 0, 'DR4': 0, 'DR5': 0, 'DR6': 0, 'DR7': 0, 'DR8': 0}

    # creating the directory that contains the data
    pathlib.Path('Dataset').mkdir(parents=True, exist_ok=True)

    azimuths = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55,
                65, 80]

    for az in azimuths:
        # creating the directory for the current azimuth
        pathlib.Path('Dataset/Az_' + str(az)).mkdir(parents=True, exist_ok=True)

        # gender mask. If 1, pick a female signal for the demographic region, else, pick a male signal
        gender_mask = np.random.choice([0, 0, 0, 0, 1, 1, 1, 1], size=8, replace=False)

        # picking one signal per demographic region
        for region_number in range(8):
            if gender_mask[region_number]:
                # pick female speaker
                all_speakers = os.listdir('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/DR' +
                                          str(region_number + 1))
                all_female = [person for person in all_speakers if person[0] == 'F']
                female_speaker = all_female[np.random.randint(low=0, high=len(all_female), size=1)[0]]

                # picking signal
                all_signals = [file for file in os.listdir('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/'
                                                           'DR' + str(region_number + 1) + '/' + female_speaker)
                               if file.endswith('_new.wav')]
                female_signal = all_signals[np.random.randint(low=0, high=len(all_signals), size=1)[0]]

                # reading the chosen signal
                fs, speech_signal = wavfile.read('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/'
                                                 'DR' + str(region_number + 1) + '/' + female_speaker + '/' +
                                                 female_signal)

                # convolving single channel signal with HRIR to obtain multi-channel signal
                multi_channel_speech = np.transpose(convolve_hrir(azimuth=az, speech_signal=speech_signal))

                np.save('Dataset/Az_' + str(az) + '/f_sig_' + str(region_number + 1) + '_Az_' + str(az) + '.npy',
                        multi_channel_speech)

                # registering information about the chosen signal
                az_info['DR' + str(region_number + 1)] = '/' + female_speaker + '/' + female_signal
            else:
                # pick male speaker
                all_speakers = os.listdir('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/DR' +
                                          str(region_number + 1))
                all_male = [person for person in all_speakers if person[0] == 'M']
                male_speaker = all_male[np.random.randint(low=0, high=len(all_male), size=1)[0]]

                # picking signal
                all_signals = [file for file in os.listdir('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/'
                                                           'DR' + str(region_number + 1) + '/' + male_speaker)
                               if file.endswith('_new.wav')]
                male_signal = all_signals[np.random.randint(low=0, high=len(all_signals), size=1)[0]]

                # reading the chosen signal
                fs, speech_signal = wavfile.read('/Users/gustavocidornelas/Desktop/sound-source/TIMIT/TRAIN/'
                                                 'DR' + str(region_number + 1) + '/' + male_speaker + '/' +
                                                 male_signal)

                # convolving single channel signal with HRIR to obtain multi-channel signal
                multi_channel_speech = np.transpose(convolve_hrir(azimuth=az, speech_signal=speech_signal))

                np.save('Dataset/Az_' + str(az) + '/m_sig_' + str(region_number + 1) + '_Az_' + str(az) + '.npy',
                        multi_channel_speech)
                # registering information about the chosen signal
                az_info['DR' + str(region_number + 1)] = '/' + male_speaker + '/' + male_signal

        # saving information about the signals used for this azimuth
        f = open('/Users/gustavocidornelas/Desktop/sound-source/Dataset/Az_' + str(az) + '/info.txt', 'w')
        f.write(str(az_info))
        f.close()
        print(0)

