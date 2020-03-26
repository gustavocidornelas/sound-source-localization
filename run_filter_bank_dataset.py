"""
Script that runs the algorithm using a linear combination of sound onset models with different frequencies on the
dataset with different speech signals (built over the TIMIT dataset). The parameters are defined at the beginning of
the script.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""

import os
import pathlib

from compute_LCR import ComputeLCR
from fit_poly import FitPoly
from delay_estimation_filter_bank import DelayEstimationFilterBank

if __name__ == '__main__':

    # defining the simulation parameters
    # onset model considered. Either 'decaying_sinusoid' or 'gammatone'
    onset_model = 'decaying_sinusoid'
    # window model considered. Either 'gamma' or 'exponential'
    window_model = 'gamma'
    # number of frequencies used simultaneously (linear combination)
    n_freq = 1
    # list of frequencies used simultaneously
    filter_bank_frequencies = [[90.0], [100.0]]  # list of lists
    # list of onset decays. Follows the same order as the list of frequencies
    onset_decays = [0.99]
    # window decay used
    window_decay = 0.999
    # list of azimuths used
    azimuth = [-80, -65, 65, 80, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
               55]

    for az in azimuth:
        # data path
        azimuth_data_path = '/Users/gustavocidornelas/Desktop/sound-source/Dataset/Az_' + str(az)

        # list of speech signals in the data path
        all_signals = [file for file in os.listdir(azimuth_data_path) if file.endswith('.npy')]

        # creating the sub-directories that contains the results (LCRs, polynomial coefficients and delay estimates)
        pathlib.Path(azimuth_data_path + '/LCR').mkdir(parents=True, exist_ok=True)
        pathlib.Path(azimuth_data_path + '/Coefficients').mkdir(parents=True, exist_ok=True)
        pathlib.Path(azimuth_data_path + '/Delays').mkdir(parents=True, exist_ok=True)
        lcr_saving_path = azimuth_data_path + '/LCR'
        coeff_saving_path = azimuth_data_path + '/Coefficients'
        delay_saving_path = azimuth_data_path + '/Delays'

        for signal in all_signals:
            # setting up all the paths and file names
            signal_id = signal[:7]
            audio_file = azimuth_data_path + '/' + signal

            # going through the frequencies in the filter bank
            for frequency in filter_bank_frequencies:
                LCR_file = lcr_saving_path + "/" + signal_id + "_" + str(onset_model) + "_" + str(onset_decays) + "_" \
                           + str(window_model) + "_" + str(window_decay) + "_Az_" + str(az) + "_freq_" \
                           + str(frequency) + ".csv"
                coeff_left_file = coeff_saving_path + '/' + signal_id + '_coeff_left_Az_' + str(az) + '_freq_' + \
                                  str(frequency) + '.csv'
                coeff_right_file = coeff_saving_path + '/' + signal_id + '_coeff_right_Az_' + str(az) + '_freq_' + \
                                   str(frequency) + '.csv'

                # computing the LCR
                lcr = ComputeLCR(onset_model=onset_model, window_model=window_model, frequencies=frequency,
                                 n_freq=n_freq, onset_decays=onset_decays, window_decay=window_decay,
                                 audio_file=audio_file, azimuth=az, signal_id=signal_id,
                                 lcr_saving_path=lcr_saving_path)
                lcr.compute_lcr()

                # fitting the polynomial to the LCRs
                fit = FitPoly(frequencies=frequency, LCR_file=LCR_file, azimuth=az, signal_id=signal_id,
                              coeff_saving_path=coeff_saving_path)
                for i in range(1, 3):
                    fit.fit_polynomial(index=i)

            # estimating the delay from the polynomial fit
            delay = DelayEstimationFilterBank(frequencies=filter_bank_frequencies, azimuth=az,
                                              delay_saving_path=delay_saving_path,
                                              coefficients_saving_path=coeff_saving_path,
                                              signal_id=signal_id)
            delay.estimate_delay()



