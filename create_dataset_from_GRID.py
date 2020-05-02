"""
Script that creates a training dataset from the GRID audiovisual sentence corpus to validate the algorithm

----------
Author: Gustavo Cid Ornelas, ETH Zurich, April 2020
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
    hrir = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/HRIR/@44.1kHz/HRIR_Az_'
                         + str(azimuth) + '.csv', delimiter=',')
    hrir_left = hrir[:, 0]
    hrir_right = hrir[:, 1]

    # convolving the impulse responses with the speech signal
    speech_left = np.convolve(hrir_left, speech_signal, mode='same')
    speech_right = np.convolve(hrir_right, speech_signal, mode='same')

    # multi-channel signal
    return np.vstack((speech_left, speech_right))


def pad_speech(speech_signal):
    """
    Function that pads a random vector (white Gaussian noise) to the beginning of the speech signal
    """
    # computing the noise variance at the beginning of the speech signal
    noise_var = np.var(speech_signal[:1000])

    pad = np.random.normal(0, np.sqrt(noise_var), 30000)  # same length as the burn-in period for the LCR (these samples are thrown away later)

    padded_speech_signal = np.hstack((pad, speech_signal))

    return padded_speech_signal


if __name__ == '__main__':
    # fixing the random number generator seed
    np.random.seed(33)

    # dictionary with the information of the files chosen per azimuth
    az_info = {'Male': [], 'Female': []}

    # creating the directory that contains the data
    pathlib.Path('Dataset from GRID').mkdir(parents=True, exist_ok=True)

    azimuths = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55,
                65, 80]

    # speaker ids
    male_speakers = ['s1', 's2', 's3', 's5', 's6']
    female_speakers = ['s4', 's7', 's11', 's15', 's16']

    for az in azimuths:
        # creating the directory for the current azimuth
        pathlib.Path('Dataset from GRID/Az_' + str(az)).mkdir(parents=True, exist_ok=True)

        # picking 10 signals: 5 from male speakers and 5 from female speakers
        for id in female_speakers:
            # picking signal
            all_signals = [file for file in os.listdir('/Users/gustavocidornelas/Desktop/sound-source/Data GRID/Female/'
                                                       + id + '/') if file.endswith('.wav')]
            female_signal = all_signals[np.random.randint(low=0, high=len(all_signals), size=1)[0]]

            # reading the chosen signal
            fs, speech_signal = wavfile.read('/Users/gustavocidornelas/Desktop/sound-source/Data GRID/'
                                             'Female/' + id + '/' + female_signal)

            # downsample speech signal to match the sampling frequency from the HRIR
            secs = speech_signal.shape[0] / float(fs)
            num_samples = int(secs * 44100.0)
            speech_signal = scipy.signal.resample(speech_signal, num_samples)

            # padding the beginning of the signal with noise
            speech_signal = pad_speech(speech_signal)

            # convolving single channel signal with HRIR to obtain multi-channel signal
            multi_channel_speech = np.transpose(convolve_hrir(azimuth=az, speech_signal=speech_signal))

            np.save('Dataset from GRID/Az_' + str(az) + '/f_sig_' + id + '_Az_' + str(az) + '.npy',
                    multi_channel_speech)

            # registering information about the chosen signal
            az_info['Female'].append(female_signal)

        for id in male_speakers:
            # picking signal
            all_signals = [file for file in os.listdir('/Users/gustavocidornelas/Desktop/sound-source/Data GRID/'
                                                       'Male/' + id) if file.endswith('.wav')]
            male_signal = all_signals[np.random.randint(low=0, high=len(all_signals), size=1)[0]]

            # reading the chosen signal
            fs, speech_signal = wavfile.read('/Users/gustavocidornelas/Desktop/sound-source/Data GRID/'
                                             'Male/' + id + '/' + male_signal)

            # downsample speech signal to match the sampling frequency from the HRIR
            secs = speech_signal.shape[0] / float(fs)
            num_samples = int(secs * 44100.0)
            speech_signal = scipy.signal.resample(speech_signal, num_samples)

            # padding the beginning of the signal with noise
            speech_signal = pad_speech(speech_signal)

            # convolving single channel signal with HRIR to obtain multi-channel signal
            multi_channel_speech = np.transpose(convolve_hrir(azimuth=az, speech_signal=speech_signal))

            np.save('Dataset from GRID/Az_' + str(az) + '/m_sig_' + id + '_Az_' + str(az) + '.npy',
                    multi_channel_speech)

            # registering information about the chosen signal
            az_info['Male'].append(male_signal)

        # saving information about the signals used for this azimuth
        f = open('/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_' + str(az) + '/info.txt', 'w')
        f.write(str(az_info))
        f.close()
