"""
Script that runs the algorithm using a linear combination of sound onset models with different frequencies. The para-
meters are defined at the beginning of the script.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""


from compute_LCR import ComputeLCR
from fit_poly import FitPoly
from delay_estimation import DelayEstimation

if __name__ == '__main__':

    # defining the simulation parameters
    # onset model considered. Ei    ther 'decaying_sinusoid' or 'gammatone'
    onset_model = 'decaying_sinusoid'
    # window model considered. Either 'gamma' or 'exponential'
    window_model = 'gamma'
    # number of frequencies used simultaneously (linear combination)
    n_freq = 1
    # list of frequencies used simultaneously
    frequencies = [80.0]
    # list of onset decays. Follows the same order as the list of frequencies
    onset_decays = [0.99]
    # window decay used
    window_decay = 0.999
    # list of azimuths used
    azimuth = [-80, -65, 65, 80, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55]

    for az in azimuth:
        # computing the LCR
        #audio_file = '/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_' + str(az) \
        #             + '.csv'
        #lcr = ComputeLCR(onset_model=onset_model, window_model=window_model, frequencies=frequencies, n_freq=n_freq,
        #                 onset_decays=onset_decays, window_decay=window_decay, audio_file=audio_file, azimuth=az)
        #lcr.compute_lcr()

        # fitting the polynomial to the LCRs
        print(az)
        #LCR_file = str(onset_model) + "_" + str(onset_decays) + "_" + str(window_model) + "_" + str(window_decay) \
        #           + "_Az_" + str(az) + "_freq_" + str(frequencies) + ".csv"
        #fit = FitPoly(frequencies=frequencies, LCR_file=LCR_file, azimuth=az)
        #for i in range(1, 3):
        #    fit.fit_polynomial(index=i)

        # estimating the delay from the polynomial fit
        coeff_left_file = 'coeff_left_Az_ ' + str(az) + '_freq_' + str(frequencies) + '.csv'
        coeff_right_file = 'coeff_right_Az_' + str(az) + '_freq_' + str(frequencies) + '.csv'
        delay = DelayEstimation(coeff_left_file=coeff_left_file, coeff_right_file=coeff_right_file,
                                frequencies=frequencies, azimuth=az)
        delay.estimate_delay()



