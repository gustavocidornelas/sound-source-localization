import os

if __name__ == "__main__":

    azimuths = [
        -80,
        -65,
        -45,
        -40,
        -35,
        -30,
        -25,
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        55,
        65,
        80,
    ]

    base_dir = "/Volumes/gco/M.Sc. Thesis/Data/Dataset from ISTS/Az_"
    # print(os.listdir(base_dir))
    # base_dir = '/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_'
    num_signals = 22

    for az in azimuths:
        print("========== Checking azimuth " + str(az) + " ==========")

        curr_dir = base_dir + str(az)

        # # checking audio signals
        # audio_signal_list = [file for file in os.listdir(curr_dir) if file.endswith('.npy')]
        # if len(audio_signal_list) == num_signals:
        #     print('Number of audio files: Check!')
        # else:
        #     print('-- Attention: ' + str(num_signals - len(audio_signal_list)) + ' audio files missing.')
        #
        # # checking LCRs
        # lcr_dir = curr_dir + '/LCR'
        # lcr_list = [file for file in os.listdir(lcr_dir) if file.endswith('.csv')]
        # if len(lcr_list) == num_signals * 5:
        #     print('Number of LCR files: Check!')
        # else:
        #     print('-- Attention: ' + str(num_signals * 5 - len(lcr_list)) + ' LCR files missing.')
        #
        # # checking polynomials
        # poly_dir = curr_dir + '/Coefficients'
        # poly_list = [file for file in os.listdir(poly_dir) if file.endswith('.csv')]
        # if len(poly_list) == num_signals * 10:
        #     print('Number of coefficients files: Check!')
        # else:
        #     print('-- Attention: ' + str(num_signals * 10 - len(poly_list)) + ' coefficient files missing.')
        #
        # # checking delays (baseline model)
        # delay_dir = curr_dir + '/Delays'
        # delay_list = [file for file in os.listdir(delay_dir) if file.endswith('.npy')]
        # if len(delay_list) == num_signals:
        #     print('Number of delays (baseline model): Check!')
        # else:
        #     print('-- Attention: ' + str(num_signals - len(delay_list)) + ' delay files missing.')

        # checking first version of confidence delays
        # first_v_conf_dir = '/Users/gustavocidornelas/Desktop/Delays - confidence first version/Az_' + str(az)
        # conf_d_list = [file for file in os.listdir(first_v_conf_dir) if file.endswith('.npy')]
        # if len(conf_d_list) == 4 * num_signals:
        #     print('Number of delays (confidence first version): Check!')
        # else:
        #     print('-- Attention: ' + str(num_signals * 4 - len(conf_d_list)) + ' delay files missing.')

        # checking integration method version of confidence delays
        int_v_conf_dir = (
            "/Volumes/gco/M.Sc. Thesis/Data/Delays - confidence integration/Az_"
            + str(az)
        )
        int_d_list = [
            file for file in os.listdir(int_v_conf_dir) if file.endswith(".npy")
        ]
        if len(int_d_list) == 4 * num_signals:
            print("Number of delays (integration method): Check!")
        else:
            print(
                "-- Attention: "
                + str(num_signals * 4 - len(int_d_list))
                + " delay files missing."
            )
