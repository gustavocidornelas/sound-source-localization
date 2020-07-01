"""
Script that runs the whole algorithm for every audio signal, saving the unique delay estimates for each azimuth

Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""
import numpy as np
import os


if __name__ == "__main__":

    # parameters
    onset_decay = 0.99
    window_decay = 0.999
    frequency = 95.0
    # coeff_left_file = '/Users/gustavocidornelas/Desktop/sound-source/Poly_Coefficients/coeff_' + \
    #                  str(float(70)) + '_LEFT.csv'
    # coeff_right_file = '/Users/gustavocidornelas/Desktop/sound-source/Poly_Coefficients/coeff_' + \
    #                   str(float(70)) + '_RIGHT.csv'

    # os.system("python delay_estimation.py " + coeff_left_file + " " + coeff_right_file + " " + str(70))

    for az in range(-90, 100, 10):
        # reading the audio signal file
        # print('Reading the audio file for the azimuth of ' + str(az) + ' degrees...')
        # audio_file = '/Users/gustavocidornelas/Desktop/sound-source/signals_female/Female_Az_' + str(az) + '.csv'

        # first step: generating the LCRs for the audio signals
        # print('Computing the LCRs...')
        # os.system("python main.py " + str(frequency) + " " + str(onset_decay) + " " + str(window_decay) + " " +
        #          audio_file + " " + str(az))

        # reading the LCR file
        # print('Reading the LCR file for the azimuth of ' + str(az) + ' degrees...')
        # LCR_file = '/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_0.99_gamma_0.999_Az_' + \
        #           str(az) + '_freq_95.0.csv'
        # locally fitting the polynomial to the left and to the right LCRs
        # for i in range(1, 3):
        #    os.system("python fit_poly_rect.py " + LCR_file + " " + str(i) + " " + str(az))

        # reading the polynomial coefficients
        print("Reading the coefficients for the azimuth of " + str(az) + " degrees...")
        coeff_left_file = (
            "/Users/gustavocidornelas/Desktop/sound-source/Male\ signal/Poly_Coefficients/coeff_"
            + str(float(az))
            + "_LEFT.csv"
        )
        coeff_right_file = (
            "/Users/gustavocidornelas/Desktop/sound-source/Male\ signal/Poly_Coefficients/coeff_"
            + str(float(az))
            + "_RIGHT.csv"
        )

        os.system(
            "python delay_estimation.py "
            + coeff_left_file
            + " "
            + coeff_right_file
            + " "
            + str(float(az))
        )
