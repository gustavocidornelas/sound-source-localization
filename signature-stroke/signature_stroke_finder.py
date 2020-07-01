from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from scipy.spatial import distance


def display_threshold_plots(first_der, second_der):
    a0 = 0
    a1 = 2.03e-6  # first_der
    a2 = -8.8e-9  #  second_der / 2
    a3 = -2.04e-10  # np.linspace(-1e-6, 1e-3, 100)

    t = np.asarray(range(-40, 40))

    poly = a0 + a1 * t + a2 * np.power(t, 2) + a3 * np.power(t, 3)
    # plt.plot(t, poly)

    a1 = 1.15e-5
    a2 = 5.19e-8

    poly = a0 + a1 * t + a2 * np.power(t, 2) + a3 * np.power(t, 3)
    plt.plot(t, poly)

    plt.show()


if __name__ == "__main__":
    # frequency list
    freq_list = ["[80.0]", "[120.0]", "[160.0]", "[200.0]", "[240.0]"]

    lcrs = dict.fromkeys(freq_list)
    # poly_coeffs = dict.fromkeys(freq_list)

    # reading the files
    full_dir = "/Users/gustavocidornelas/Desktop/sound-source/signature-stroke"
    speech = np.load(full_dir + "/ists_sig_2_az_-30.npy")
    with open(full_dir + "/Delays/delay_info.pkl", "rb") as handle:
        delay_info = pickle.load(handle)
    use_delay_info = delay_info["ists_sig_2"]

    # display_threshold_plots(first_der=2e-6, second_der=-8e-10)

    poly_coeff_left = np.genfromtxt(
        full_dir + "/Coefficients/ists_sig_2_coeff_left_Az_-30_freq_[120.0].csv",
        skip_header=False,
        delimiter=",",
    )
    poly_coeff_right = np.genfromtxt(
        full_dir + "/Coefficients/ists_sig_2_coeff_right_Az_-30_freq_[120.0].csv",
        skip_header=False,
        delimiter=",",
    )

    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('a0')
    # ax.set_ylabel('a1')
    # ax.set_zlabel('a2')

    t = np.linspace(-100, 100, 500).reshape(500, 1)
    powers = np.hstack((np.ones((500, 1)), t, np.power(t, 2), np.power(t, 3)))
    polys_plot_left = poly_coeff_left @ np.transpose(powers)
    polys_plot_right = poly_coeff_right @ np.transpose(powers)

    # sliding window length
    window_half_len = 50

    offset = 0  # 120000 - 100
    delay_estimates = np.zeros(poly_coeff_left.shape[0])
    instant_velocity_left = np.zeros(poly_coeff_left.shape[0])
    instant_velocity_right = np.zeros(poly_coeff_left.shape[0])
    acc_left = np.zeros(poly_coeff_left.shape[0])
    acc_right = np.zeros(poly_coeff_left.shape[0])
    confidence = np.zeros(poly_coeff_left.shape[0])

    # A = np.load('delays_carai.npy')
    # plt.plot(range(A.shape[0]), A)
    # plt.show()

    plt.figure()
    for k in range(
        window_half_len + offset, poly_coeff_left.shape[0] + offset - window_half_len
    ):
        coeff_left_window = poly_coeff_left[
            k - window_half_len : k + window_half_len, :
        ]
        coeff_right_window = poly_coeff_right[
            k - window_half_len : k + window_half_len, :
        ]

        instant_velocity_left[k] = np.linalg.norm(
            coeff_left_window[-1, :] - coeff_left_window[-2, :]
        )
        instant_velocity_right[k] = np.linalg.norm(
            coeff_right_window[-1, :] - coeff_right_window[-2, :]
        )

        acc_left[k] = np.linalg.norm(
            instant_velocity_left[k] - instant_velocity_left[k - 1]
        )
        acc_right[k] = np.linalg.norm(
            instant_velocity_right[k] - instant_velocity_right[k - 1]
        )

        # computing all the distances between polynomials within the window
        distances = distance.cdist(coeff_left_window, coeff_right_window, "euclidean")

        # finding the two most similar polynomials inside the window
        id_min_poly_left = np.where(distances == distances.min())[0][0]
        id_min_poly_right = np.where(distances == distances.min())[1][0]

        # retrieving the corresponding coefficients
        coeff_min_left = coeff_left_window[id_min_poly_left, :]
        coeff_min_right = coeff_right_window[id_min_poly_right, :]

        # checking some conditions
        condition_1 = coeff_min_left[0] > 0 and coeff_min_right[0] > 0
        condition_2 = coeff_min_left[1] > 1.2e-6 and coeff_min_right[1] > 1.2e-6
        condition_3 = 2 * coeff_min_left[2] < 1e-12 and 2 * coeff_min_right[2] < 1e-12

        if condition_1 and condition_2 and condition_3:
            delay_estimates[k] = abs(
                np.where(distances == distances.min())[0][0]
                - np.where(distances == distances.min())[1][0]
            )
            confidence[k] = 1 / distances.min()
            if k == 14531:
                plt.figure()
                plt.plot(t, polys_plot_left[k, :])
                plt.plot(t, polys_plot_right[k, :])
                plt.show()

            print(k)
            print("Left sample: " + str(np.where(distances == distances.min())[0][0]))
            print("Right sample: " + str(np.where(distances == distances.min())[1][0]))
            print(delay_estimates[k])
            print(confidence[k])

        # good region condition
        # condition_1 = poly_coeff_left[k+offset, 0] > 0 and poly_coeff_right[k+offset, 0] > 0
        # condition_2 = poly_coeff_left[k+offset, 1] > 1.2e-6 and poly_coeff_right[k+offset, 1] > 1.2e-6
        # condition_3 = np.logical_and(2 * poly_coeff_left[k+offset, 2] < 1e-13, 2 * poly_coeff_right[k+offset, 2] > 1e-13)

        # good_region = condition_1 and condition_2
        # good_region = np.logical_and(good_region, condition_3)

        # if good_region:
        #     print('Good region !!!' + str(k))
        #     plt.title('Good region !!!' + str(k))
        # else:
        #     #print('Bad region...')
        #     plt.title('Bad...' + str(k))

        sns.heatmap(distances, cbar=False)
        plt.pause(0.05)
        plt.cla()
        #
        # plt.plot(t, polys_plot_left[k+offset, :])
        # plt.pause(0.05)
        # print(0)
    # np.save('delays_carai.npy', delay_estimates)

    # plt.figure()
    # plt.plot(range(poly_coeff_left.shape[0]), instant_velocity_left)
    # plt.plot(range(poly_coeff_left.shape[0]), instant_velocity_right)
    #
    # plt.figure()
    # plt.plot(range(poly_coeff_left.shape[0]), acc_left)
    # plt.plot(range(poly_coeff_left.shape[0]), acc_right)
    plt.figure()
    plt.plot(range(poly_coeff_left.shape[0]), delay_estimates)

    plt.figure()
    plt.plot(range(poly_coeff_left.shape[0]), poly_coeff_left[:, 0])

    plt.figure()
    plt.plot(range(poly_coeff_left.shape[0]), confidence)

    plt.show()

    # standardizing withing the window
    # coeff_left_window[:, 0] = (coeff_left_window[:, 0] - np.mean(coeff_left_window[:, 0])) / np.std(
    #     coeff_left_window[:, 0])
    # coeff_left_window[:, 1] = (coeff_left_window[:, 1] - np.mean(coeff_left_window[:, 1])) / np.std(
    #     coeff_left_window[:, 1])
    # coeff_left_window[:, 2] = (coeff_left_window[:, 2] - np.mean(coeff_left_window[:, 2])) / np.std(
    #     coeff_left_window[:, 2])
    #
    # coeff_right_window[:, 0] = (coeff_right_window[:, 0] - np.mean(coeff_right_window[:, 0])) / np.std(
    #     coeff_right_window[:, 0])
    # coeff_right_window[:, 1] = (coeff_right_window[:, 1] - np.mean(coeff_right_window[:, 1])) / np.std(
    #     coeff_right_window[:, 1])
    # coeff_right_window[:, 2] = (coeff_right_window[:, 2] - np.mean(coeff_right_window[:, 2])) / np.std(
    #     coeff_right_window[:, 2])

    # ax_polys.plot(np.vstack([np.transpose(t)] * (2 * window_half_len)), polys_plot_left[k - window_half_len: k + window_half_len, :])

    # plotting trajectory
    # ax.scatter(coeff_left_window[::2, 0], coeff_left_window[::2, 1],
    #            coeff_left_window[::2, 2])
    # ax.scatter3D(coeff_right_window[::2, 0], coeff_right_window[::2, 1],
    #              coeff_right_window[::2, 2])
    # plt.pause(0.05)
    # plt.cla()
    # print(k)

    # # defining masks
    # a0_mask_left = poly_coeff_left[:, 0] > 0
    # a1_mask_left = poly_coeff_left[:, 1] > 0
    # left_mask = np.logical_and(a0_mask_left, a1_mask_left)
    # a0_mask_right = poly_coeff_right[:, 0] > 0
    # a1_mask_right = poly_coeff_right[:, 1] > 0
    # right_mask = np.logical_and(a0_mask_right, a1_mask_right)
    #
    # # standardizing coefficients
    # poly_coeff_left[:, 0] = (poly_coeff_left[:, 0] - np.mean(poly_coeff_left[:, 0])) / np.std(
    #     poly_coeff_left[:, 0])
    # poly_coeff_left[:, 1] = (poly_coeff_left[:, 1] - np.mean(poly_coeff_left[:, 1])) / np.std(
    #     poly_coeff_left[:, 1])
    # poly_coeff_left[:, 2] = (poly_coeff_left[:, 2] - np.mean(poly_coeff_left[:, 2])) / np.std(
    #     poly_coeff_left[:, 2])
    #
    # poly_coeff_right[:, 0] = (poly_coeff_right[:, 0] - np.mean(poly_coeff_right[:, 0])) / np.std(
    #     poly_coeff_right[:, 0])
    # poly_coeff_right[:, 1] = (poly_coeff_right[:, 1] - np.mean(poly_coeff_right[:, 1])) / np.std(
    #     poly_coeff_right[:, 1])
    # poly_coeff_right[:, 2] = (poly_coeff_right[:, 2] - np.mean(poly_coeff_right[:, 2])) / np.std(
    #     poly_coeff_right[:, 2])
    #
    # # selecting rows with the mask
    # # poly_coeff_left = poly_coeff_left[left_mask, :]
    # # poly_coeff_right = poly_coeff_right[right_mask, :]
    #
    # num_samples_plot = int(27380 / 15)
    # low_lim = 13000
    # high_lim = 15000
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('a0')
    # ax.set_ylabel('a1')
    # ax.set_zlabel('a2')
    # ax.scatter(poly_coeff_left[::5, 0], poly_coeff_left[::5, 1],
    #            poly_coeff_left[::5, 2])
    # ax.scatter3D(poly_coeff_right[::5, 0], poly_coeff_right[::5, 1],
    #              poly_coeff_right[::5, 2])
    # plt.show()
    #
    # print(poly_coeff_left[14261, :])
    # print(poly_coeff_right[14261, :])
    #
    # #fig, axs = plt.subplots(len(freq_list), sharex=True)
    # for i, freq in enumerate(freq_list):
    #     # loading the LCRs
    #     lcrs[freq] = np.genfromtxt(full_dir + '/LCR/ists_sig_2_decaying_sinusoid_[0.99]_gamma_0.999_Az_-30_freq_' +
    #                                freq + '.csv', skip_header=True, delimiter=',')
    #     lcrs[freq] = lcrs[freq][30000:, :]
    #     # axs[i].plot(range(lcrs[freq].shape[0]), lcrs[freq][:, 1])
    #     # axs[i].plot(range(lcrs[freq].shape[0]), lcrs[freq][:, 2])
    #     # axs[i].set_ylim([0, 0.0025])
    #
    #     #if freq == str(use_delay_info[0]):
    #         # axs[i].axvline(x=use_delay_info[1], linestyle='--', color='k')
