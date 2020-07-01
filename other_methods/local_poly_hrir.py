"""
Estimation of the ITD by locally fitting a 3rd degree polynomial to the HRIR with a rectangular window and then
estimating their delay with the same delay estimation procedure done by our algorithm

----------
Author: Gustavo Cid Ornelas, ETH Zurich, March 2020
"""
import numpy as np
import matplotlib.pyplot as plt


def local_poly_fit(signal):
    # azimuth list
    azimuth = [
        -80,
        -65,
        -55,
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

    # defining the rectangular window parameters
    n_window = 1
    window_decay = 1
    length_window = 51
    a = int((1 - length_window) / 2)  # beginning of the window
    b = int((length_window - 1) / 2)  # end of the window

    # loading the 3rd degree polynomial model
    n_states = 4
    A_poly = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    A_poly_inv = np.linalg.inv(A_poly)
    s_poly = np.array([[0], [0], [0], [1]])
    # some useful products used in the recursions
    A_a_s = np.dot(np.linalg.matrix_power(A_poly, a), s_poly)
    A_b_1_s = np.dot(np.linalg.matrix_power(A_poly, b + 1), s_poly)

    for index in range(2):
        hrir = signal[:, index]

        # signal length
        L = hrir.shape[0]

        if index == 0:
            print("Fitting the polynomial to the HRIR from the left...")
        elif index == 1:
            print("Fitting the polynomial to the HRIR from the right...")

        # initializing the other messages
        k_k_f = 0
        k_k_b = 0
        xi_k_f = np.zeros((n_states, 1))
        xi_k_b = np.zeros((n_states, 1))
        W_k_f = np.zeros((n_states, n_states))
        W_k_b = np.zeros((n_states, n_states))

        # sliding the rectangular window through the signal
        # starting at the initial position
        # forward pass
        for k in range(a, 0):
            k_k_f += np.power(hrir[k - a], 2)
            xi_k_f += np.dot(np.linalg.matrix_power(A_poly, k), s_poly) * hrir[k - a]
            W_k_f += np.linalg.multi_dot(
                [
                    np.linalg.matrix_power(A_poly_inv, -k),
                    s_poly,
                    np.transpose(s_poly),
                    np.transpose(np.linalg.matrix_power(A_poly_inv, -k)),
                ]
            )

        # backward pass
        for k in range(0, b + 1):
            k_k_b += np.power(hrir[k + b], 2)
            xi_k_b += np.dot(np.linalg.matrix_power(A_poly, k), s_poly) * hrir[k + b]
            W_k_b += np.linalg.multi_dot(
                [
                    np.linalg.matrix_power(A_poly, k),
                    s_poly,
                    np.transpose(s_poly),
                    np.transpose(np.linalg.matrix_power(A_poly, k)),
                ]
            )

        # combining the two messages
        W_k = W_k_f + W_k_b
        xi_k = xi_k_f + xi_k_b
        W_k_inv = np.linalg.inv(W_k)
        # calculating the optimal C
        C = np.transpose(np.dot(W_k_inv, xi_k))

        # message recursive updates for the other window positions
        C_opt = np.zeros((L + a - 2 + b + 1, 4))
        for k0 in range(b + 1, L + a - 1):
            # forward pass
            k_k_f = k_k_f - np.power(hrir[k0 - 1 + a], 2) + np.power(hrir[k0 - 1], 2)
            xi_k_f = np.dot(
                A_poly_inv, xi_k_f - A_a_s * hrir[k0 - 1 + a] + s_poly * hrir[k0 - 1]
            )

            # backward pass
            k_k_b = k_k_b - np.power(hrir[k0 - 1], 2) + np.power(hrir[k0 + b], 2)
            xi_k_b = np.dot(
                A_poly_inv, xi_k_b - s_poly * hrir[k0 - 1] + A_b_1_s * hrir[k0 + b]
            )

            # combining the two messages
            xi_k = xi_k_f + xi_k_b
            k_k = k_k_f + k_k_b

            # calculating the optimal C
            C = np.transpose(np.dot(W_k_inv, xi_k))

            C_opt[k0, :] = np.dot(
                C,
                np.array(
                    [
                        [0, 1 / 3.0, -0.5, 1 / 6.0],
                        [0, -0.5, 0.5, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                    ]
                ),
            )

        if index == 0:
            np.savetxt("signal_left.csv", C_opt, delimiter=",")
        elif index == 1:
            np.savetxt("signal_right.csv", C_opt, delimiter=",")


def delay_estimation():
    # azimuth list
    azimuth = [
        -80,
        -65,
        -55,
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
    plt.figure()
    for az in azimuth:

        print("Estimating the delays for azimuth " + str(az))

        coeff_left = np.genfromtxt(
            "/Users/gustavocidornelas/Desktop/sound-source/Other methods/coeff_HRIR_left_Az_"
            + str(az)
            + ".csv",
            delimiter=",",
            skip_header=False,
        )
        coeff_right = np.genfromtxt(
            "/Users/gustavocidornelas/Desktop/sound-source/Other methods/coeff_HRIR_right_Az_"
            + str(az)
            + ".csv",
            delimiter=",",
            skip_header=False,
        )

        left_peak = 0
        right_peak = 0

        # if az == 80:
        #    plt.figure()
        #    plt.plot(range(199), coeff_left[:, 0])
        #    plt.plot(range(199), coeff_right[:, 0])
        #    plt.show()

        # main loop, going through each sample coefficients
        for k in range(coeff_left.shape[0]):
            # retrieving the current sample's coefficients
            coeff_left_k = coeff_left[k, :]
            coeff_right_k = coeff_right[k, :]

            # if k == 50 and az == 80:
            #    x = np.linspace(-10, 10)
            #    y = coeff_left_k[0] + coeff_left_k[1] * x + coeff_left_k[2] * np.power(x, 2) + coeff_left_k[3] * np.power(x, 3)
            #    print(coeff_left_k)
            #    plt.figure()
            #    plt.plot(x, y)
            #    plt.plot(range(-10, 10), coeff_left[40:60, 0])
            #    plt.show()

            # looking for the first peak
            if (
                abs(coeff_left_k[1]) < 4e-4
                and 2 * coeff_left_k[2] < 0
                and left_peak == 0
            ):
                left_peak = k
                print("Left!")
            if (
                abs(coeff_right_k[1]) < 4e-4
                and 2 * coeff_right_k[2] < 0
                and right_peak == 0
            ):
                right_peak = k
                print("Right!")

        delay = abs(left_peak - right_peak)
        if az == 80:
            print(left_peak)
            print(right_peak)

        # np.save('teste_all_delays_Az_' + str(azimuth) + '.npy', total_delay)
        # print(np.unique(total_delay)/44.1)
        print(delay / 44.1)
        # np.savetxt('unique_delays_HRIR_Az_' + str(az) + '.csv', np.unique(total_delay) / 44.1, delimiter=',')
        plt.scatter(az, delay / 44.1)


if __name__ == "__main__":
    audio = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_30.csv",
        delimiter=",",
    )
    audio_onset = audio[238000:238600, :]

    local_poly_fit(audio_onset)

    plt.figure()
    plt.plot(range(audio_onset.shape[0]), audio_onset[:, 0])
    plt.plot(range(audio_onset.shape[0]), audio_onset[:, 1])
    plt.show()

    # local_poly_fit()
    # delay_estimation()
    # plt.show()
