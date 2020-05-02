import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import entropy
from tabulate import tabulate
from textwrap import fill

class InfoLog:

    def __init__(self, azimuth):
        self.azimuth = azimuth
        self.not_sparse = []
        self.no_delays = []
        self.off_delays = []

    def check_sparse(self, signal_id, lcr_file):
        # opening the LCR file
        lcr = np.genfromtxt(lcr_file, delimiter=',', skip_header=True)

        # retrieving the frequency
        freq = lcr_file.split('freq_')[1][:-4]

        # entropy threshold
        thresh = 9.45

        # checking if LCR is sparse
        if entropy(lcr[30000:, 0]) > thresh or entropy(lcr[30000:, 1]) > thresh:
            self.not_sparse.append('(' + signal_id + ', @' + freq + ' Hz)')

    def check_no_delays(self, signal_id, all_delays):

        if all(all_delays == np.zeros(all_delays.shape)):
            self.no_delays.append(signal_id)

    def check_off_delays(self, signal_id, all_delays):

        # delay threshold (in samples)
        thresh = 1.5 * 44.1

        if any(np.unique(all_delays) > thresh):
            item_index = np.where(all_delays == np.max(np.unique(all_delays)))[0][0]
            self.off_delays.append('(' + signal_id + ', k = ' + str(item_index) + ')')

    def display_table(self, obj_list):

        # list with the elements from the table
        table_list = []

        # going through the object list
        for obj in obj_list:
            # zipping together the information for the current table entry
            line_lim = 3
            az_entry = []
            az_entry.append(obj.azimuth)
            az_entry.append(fill(str(obj.not_sparse), width=80))
            az_entry.append(fill(str(obj.no_delays), width=80))
            az_entry.append(fill(str(obj.off_delays), width=80))
            table_list.append(az_entry)

        print(tabulate(table_list, headers=['Azimuth', 'Not sparse', 'No delays', 'Weird delays'], tablefmt='fancy_grid'))

    def visualize_unique_delays(self, azimuths):
        # sampling rate
        fs = 44100.0

        # dictionary that stores all the unique delays for each azimuth
        all_delays_dict = dict.fromkeys(azimuths, [])
        medians = []

        for az in azimuths:

            # directory with all the delay files
            dir_delay_files = '/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_' + str(
                az) + '/Delays'

            # listing all the files within the directory
            all_delay_files = [file for file in os.listdir(dir_delay_files) if file.endswith('.npy')]

            for file in all_delay_files:
                delays = np.load(dir_delay_files + '/' + file)
                unique_delays = np.unique(delays)

                # plt.scatter(az, np.median(unique_delays[1:]) * 1000 / fs, label=file.split('_all')[0])
                teste_del = unique_delays[1:]
                #plt.scatter(np.repeat(az, unique_delays.shape[0] - 1), unique_delays[1:] * 1000 / fs)
                #plt.scatter(az, np.min(unique_delays[1:]) * 1000 / fs)
                # saving the unique delays for that azimuth
                all_delays_dict[az].extend(unique_delays[1:])

            #medians.append(np.median(unique_delays) * 1000 / fs)

        # calculating the theoretical delays for comparison

        theo_delays = [(152/343) * np.sin(np.deg2rad(80.0)), (152/343) * np.sin(np.deg2rad(65.0)),
                       (152 / 343) * np.sin(np.deg2rad(55.0)), (152/343) * np.sin(np.deg2rad(45.0)),
                       (152/343) * np.sin(np.deg2rad(35.0)), (152/343) * np.sin(np.deg2rad(25.0)),
                       (152/343) * np.sin(np.deg2rad(15.0)), (152/343) * np.sin(np.deg2rad(5.0)),
                       (152 / 343) * np.sin(np.deg2rad(5.0)), (152 / 343) * np.sin(np.deg2rad(15.0)),
                       (152 / 343) * np.sin(np.deg2rad(25.0)), (152 / 343) * np.sin(np.deg2rad(35.0)),
                       (152 / 343) * np.sin(np.deg2rad(45.0)), (152 / 343) * np.sin(np.deg2rad(55.0)),
                       (152 / 343) * np.sin(np.deg2rad(65.0)), (152 / 343) * np.sin(np.deg2rad(80.0))]


        # plt.legend()
        # plt.ylim([0, 2.5])'
        plt.xlabel('Azimuth [deg]')
        plt.ylabel('ITD [msec]')

        # plotting some reference values
        #plt.plot(azimuths, medians)
        plt.plot([-80, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 80], theo_delays)
        plt.show()


if __name__ == '__main__':
    azimuths = [80, 65, 55, 45, 35, 25, 15, 5, -80, -65, -55, -45, -35, -25, -15, -5]

    p = InfoLog(azimuths[0])
    p.visualize_unique_delays(azimuths)