"""
Script to visualize the speech signal, corresponding LCRs and delay estimates for the signals on the speech dataset
(built over the TIMIT dataset). Displays a single figure with multiple subplots.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun


def display_plots(azimuth, signal_id, fs=44100.0, burn_in=30000):

    # setting up the paths
    azimuth_data_path = '/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_' + str(azimuth)
    lcr_path = azimuth_data_path + '/LCR'
    coeff_path = azimuth_data_path + '/Coefficients'
    delay_path = azimuth_data_path + '/Delays'

    # retrieving results for the specified signal
    audio_signal = np.load(azimuth_data_path + '/' + signal_id + '_Az_' + str(azimuth) + '.npy')
    audio_signal = audio_signal[burn_in:, :]
    all_lcr_files = [file for file in os.listdir(lcr_path) if file.startswith(signal_id)]
    all_delay_files = [file for file in os.listdir(delay_path) if file.startswith(signal_id)]
    all_coeff_files = [file for file in os.listdir(coeff_path) if file.startswith(signal_id)]

    # loading the dict with the delay info
    delay_info = pickle.load(open(delay_path + '/delay_info.pkl', "rb"))
    curr_delay_info = delay_info[signal_id]

    # creating the plot
    #plt.figure()
    #plt.plot(np.asarray(range(audio_signal.shape[0])), audio_signal[:, 0], 'b')
    #plt.plot(np.asarray(range(audio_signal.shape[0])), audio_signal[:, 1], 'r')
    # plt.axvline(x=33149, linestyle='--', color='k')

    fig, axs = plt.subplots(2 + len(all_lcr_files), figsize=(20, 7), sharex=True)  # one for the audio signal, one for the delays and all the LCRs
    # fig, axs = plt.subplots(len(all_lcr_files))
    axs[0].set_title(signal_id + ' @' + str(azimuth) + ' deg')
    axs[0].plot(np.asarray(range(audio_signal.shape[0])), audio_signal[:, 0], 'b')
    axs[0].plot(np.asarray(range(audio_signal.shape[0])), audio_signal[:, 1], 'r')

    # plotting the LCRs
    for i, lcr_file in enumerate(all_lcr_files):
        lcr = np.genfromtxt(lcr_path + '/' + lcr_file, delimiter=',', skip_header=True)
        lcr = lcr[burn_in:, :]

        # onset frequency to generate this LCR
        freq = lcr_file.split('_freq_', 1)[1][:-4]

        if freq == str(curr_delay_info[0]):
            axs[i + 1].axvline(x=curr_delay_info[1], linestyle='--', color='k')

        # plotting
        axs[i + 1].plot(np.asarray(range(lcr.shape[0])), lcr[:, 1], 'b')
        axs[i + 1].plot(np.asarray(range(lcr.shape[0])), lcr[:, 2], 'r')
        axs[i + 1].set_title(freq + ' Hz')
        axs[i + 1].set_ylim([0, 0.0025])

    # plotting the delays
    delays = np.load(delay_path + '/' + all_delay_files[0])
    axs[-1].plot(np.asarray(range(delays.shape[0])), delays * 1000 / fs, 'k')  # plot delay in msec
    axs[-1].set_xlabel('Time [s]')
    scale = 1.5
    f = zoom_factory(axs[1], base_scale=scale)
    plt.draw()
    plt.waitforbuttonpress()
    plt.cla()


def display_threshold_plots(first_der, second_der, free_param):
    a0 = 0
    a1 = first_der
    a2 = second_der / 2
    a3 = free_param

    t = np.asarray(range(-40, 40))

    a1_values = [1.88e-7, 5e-7, 1.2e-6, 3e-6]

    for a1 in a1_values:

        poly = a0 + a1 * t + a2 * np.power(t, 2) + a3 * np.power(t, 3)

        plt.plot(t, poly, label=str(a1))


    plt.legend()
    plt.show()


if __name__ == '__main__':
    # sampling rate
    fs = 44100.0
    # burn-in period
    burn_in = 30000

    # visualize results for the following azimuth and the following speech signal
    azimuths = [5, 15, 25, 35, 45, 55, 65, 80] #[-80, -65, -55, -45,
    signal_ids = ['f_sig_s4', 'f_sig_s7', 'f_sig_s11', 'f_sig_s15', 'f_sig_s16',
                  'm_sig_s1', 'm_sig_s2', 'm_sig_s3', 'm_sig_s5', 'm_sig_s6']

    for az in azimuths:
        for signal in signal_ids:
            display_plots(az, signal)
    
    #display_threshold_plots(1.2e-6, 1e-13, -2.6e-10)