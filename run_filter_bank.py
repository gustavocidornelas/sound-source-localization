"""
Script that runs the algorithm using a filter bank (i.e., using multiple onset model frequencies in parallel). The para-
meters are defined at the beginning of the script.

----------
Author: Gustavo Cid Ornelas, ETH Zurich, February 2020
"""


from compute_LCR import ComputeLCR
from fit_poly import FitPoly
from delay_estimation_filter_bank import DelayEstimationFilterBank

if __name__ == '__main__':

    # defining the simulation parameters
    # onset model considered. Either 'decaying_sinusoid' or 'gammatone'
    onset_model = 'decaying_sinusoid'
    # window model considered. Either 'gamma' or 'exponential'
    window_model = 'gamma'
    # number of frequencies used simultaneously (linear combination). In this case, should always be 1
    n_freq = 1
    # list of frequencies that comprise the filter bank
    filter_bank_frequencies = [[90.0], [100.0]]  # list of lists
    # list of onset decays used, in this case, should always contain a single element
    onset_decays = [0.99]
    # window decay used
    window_decay = 0.999
    # list o azimuths
    azimuth = [90]

    for az in azimuth:
        for frequency in filter_bank_frequencies:
            # computing the LCR
            audio_file = '/Users/gustavocidornelas/Desktop/sound-source/Male signal/Audio_signals/Male_Az_' + str(az) \
                         + '.csv'
            lcr = ComputeLCR(onset_model=onset_model, window_model=window_model, frequencies=frequency, n_freq=n_freq,
                             onset_decays=onset_decays, window_decay=window_decay, audio_file=audio_file, azimuth=az)
            lcr.compute_lcr()

            # fitting the polynomial to the LCRs
            LCR_file = str(onset_model) + "_" + str(onset_decays) + "_" + str(window_model) + "_" + str(window_decay) \
                    + "_Az_" + str(az) + "_freq_" + str(frequency) + ".csv"
            fit = FitPoly(frequencies=frequency, LCR_file=LCR_file, azimuth=az)
            for i in range(1, 3):
                fit.fit_polynomial(index=i)

        # estimating the delay from the polynomial fit
        delay = DelayEstimationFilterBank(frequencies=frequency, azimuth=az)
        delay.estimate_delay()
