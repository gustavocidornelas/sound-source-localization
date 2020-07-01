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
        lcr = np.genfromtxt(lcr_file, delimiter=",", skip_header=True)

        # retrieving the frequency
        freq = lcr_file.split("freq_")[1][:-4]

        # entropy threshold
        thresh = 9.45

        # checking if LCR is sparse
        if entropy(lcr[30000:, 0]) > thresh or entropy(lcr[30000:, 1]) > thresh:
            self.not_sparse.append("(" + signal_id + ", @" + freq + " Hz)")

    def check_no_delays(self, signal_id, all_delays):

        if all(all_delays == np.zeros(all_delays.shape)):
            self.no_delays.append(signal_id)

    def check_off_delays(self, signal_id, all_delays):

        # delay threshold (in samples)
        thresh = 1.5 * 44.1

        if any(np.unique(all_delays) > thresh):
            item_index = np.where(all_delays == np.max(np.unique(all_delays)))[0][0]
            self.off_delays.append("(" + signal_id + ", k = " + str(item_index) + ")")

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

        print(
            tabulate(
                table_list,
                headers=["Azimuth", "Not sparse", "No delays", "Weird delays"],
                tablefmt="fancy_grid",
            )
        )

    def visualize_unique_delays(self, azimuths):
        # sampling rate
        fs = 44100.0

        # dictionary that stores all the unique delays for each azimuth
        means = np.zeros(len(azimuths))
        medians = np.zeros(len(azimuths))

        # for the poly fit
        x = []
        y = []

        for i, az in enumerate(azimuths):
            sum_tot = 0
            num_signals = 0
            current_delays = []

            # directory with all the delay files
            dir_delay_files = (
                "/Users/gustavocidornelas/Desktop/sound-source/Dataset from ISTS/Az_"
                + str(az)
                + "/Delays"
            )

            dir_conf_1 = (
                "/Users/gustavocidornelas/Desktop/Delays - confidence first version/Az_"
                + str(az)
            )
            new_del_1 = []
            dir_conf_2 = (
                "/Users/gustavocidornelas/Desktop/Delays - confidence integration/Az_"
                + str(az)
            )
            new_del_2 = []

            # listing all the files within the directory
            all_delay_files = [
                file for file in os.listdir(dir_delay_files) if file.endswith(".npy")
            ]

            for file in all_delay_files:
                if file[0] == "i" and file.split("sig_")[1].split("_all")[0] not in [
                    "3",
                    "7",
                    "12",
                    "19",
                ]:  # only signals without fricative
                    delays = np.load(dir_delay_files + "/" + file)
                    unique_delays = np.unique(delays)

                    try:
                        new_delays_1 = np.load(dir_conf_1 + "/" + file)
                        new_del_1.append(np.median(new_delays_1))

                        new_delays_2 = np.load(dir_conf_2 + "/" + file)
                        new_del_2.append(np.median(new_delays_2))
                    except Exception:
                        print(0)

                    # plt.scatter(az, np.median(unique_delays[1:]) * 1000 / fs, label=file.split('_all')[0])
                    teste_del = unique_delays[1:]
                    # plt.scatter(np.repeat(az, unique_delays.shape[0] - 1), unique_delays[1:] * 1000 / fs)
                    # plt.scatter(az, np.min(unique_delays[1:]) * 1000 / fs)
                    # saving the unique delays for that azimuth
                    sum_tot = sum_tot + unique_delays[-1]
                    num_signals = num_signals + 1
                    current_delays.append(unique_delays[-1])
                    x.append(az)
                    y.append(unique_delays[-1])

            means[i] = (sum_tot / num_signals) * 1000 / fs
            medians[i] = np.median(current_delays) * 1000 / fs
            if az == 0:
                plt.errorbar(
                    x=az,
                    y=means[i],
                    yerr=np.std(current_delays) * 1000 / fs,
                    fmt="go",
                    ecolor="g",
                    capsize=3,
                    label="Our first version",
                )
            #     # plt.errorbar(x=az, y=np.mean(new_del_1) * 1000 / 44100.0, yerr=np.std(new_del_1) * 1000 / 44100.0,
            #     #             fmt='bo', ecolor='b', capsize=3, label='Conf. first')
            #     plt.errorbar(x=az, y=np.mean(new_del_2) * 1000 / 44100.0, yerr=np.std(new_del_2) * 1000 / 44100.0,
            #                  fmt='go', ecolor='g', capsize=3, label='Conf. integration')
            else:
                plt.errorbar(
                    x=az,
                    y=means[i],
                    yerr=np.std(current_delays) * 1000 / fs,
                    fmt="go",
                    ecolor="g",
                    capsize=3,
                )
            #     # plt.errorbar(x=az, y=np.mean(new_del_1) * 1000 / 44100.0, yerr=np.std(new_del_1) * 1000 / 44100.0,
            #     #              fmt='bo', ecolor='b', capsize=3)
            #     plt.errorbar(x=az, y=np.mean(new_del_2) * 1000 / 44100.0, yerr=np.std(new_del_2) * 1000 / 44100.0,
            #                  fmt='go', ecolor='g', capsize=3)

        # fitting a 3rd degree polynomial to the data
        x = np.asarray(x)
        y = np.asarray(y) * 1000 / fs
        poly = np.polyfit(x, y, 3)
        x_plot = np.linspace(-55, 80, 100)
        y_plot = (
            poly[3]
            + poly[2] * x_plot
            + poly[1] * np.power(x_plot, 2)
            + poly[0] * np.power(x_plot, 3)
        )

        # plt.plot(x_plot, y_plot, 'k', label='3rd degree LS fit +/- 1 std')

        # calculating the theoretical delays for comparison
        az_plot = np.asarray(
            [
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
                -5,
                -10,
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
                65,
                80,
            ]
        )
        theo_delays = (152 / 343) * np.sin(np.deg2rad(az_plot))

        # plt.legend()
        # plt.ylim([0, 2.5])'
        plt.xlabel("Azimuth [deg]")
        plt.ylabel("ITD [msec]")
        # plotting some reference values
        # plt.plot(np.asarray(azimuths), medians)
        # plt.plot(np.asarray(azimuths), means)
        # plt.plot(az_plot, abs(theo_delays), label='Theoretical delays')
        plt.legend()

    def visualize_other_methods_delays(self, azimuths):

        for i, az in enumerate(azimuths):

            # loading delays
            thresh_delays = np.load(
                "/Users/gustavocidornelas/Desktop/sound-source/Dataset from ISTS/Az_"
                + str(az)
                + "/Delays threshold method/ists_threshold_delays_Az_"
                + str(az)
                + ".npy"
            )
            cross_corr_delays = np.load(
                "/Users/gustavocidornelas/Desktop/sound-source/Dataset from ISTS/Az_"
                + str(az)
                + "/Delays cross-correlation method/ists_cross_corr_delays_Az_"
                + str(az)
                + ".npy"
            )
            cross_corr_frame_delays = abs(
                np.load(
                    "/Users/gustavocidornelas/Desktop/sound-source/mini-world/Dataset from ISTS/Az_"
                    + str(az)
                    + "/Delays cross-correlation per frame method/ists_cross_corr_frame_delays_Az_"
                    + str(az)
                    + ".npy"
                )
            )

            semi_cross_corr_frame_delays = abs(
                np.load(
                    "/Users/gustavocidornelas/Desktop/sound-source/mini-world/Dataset from ISTS/Az_"
                    + str(az)
                    + "/Delays semi cross-correlation per frame method/ists_cross_corr_frame_delays_Az_"
                    + str(az)
                    + ".npy"
                )
            )

            old_cross_corr_frame_delays = abs(
                np.load(
                    "/Users/gustavocidornelas/Desktop/sound-source/Dataset from ISTS/Az_"
                    + str(az)
                    + "/Delays cross-correlation per frame method/ists_cross_corr_frame_delays_Az_"
                    + str(az)
                    + ".npy"
                )
            )

            our_new_delays = abs(
                np.load(
                    "/Users/gustavocidornelas/Desktop/sound-source/mini-world/Dataset from ISTS/Az_"
                    + str(az)
                    + "/Delays comp. our method/ists_our_delays_Az_"
                    + str(az)
                    + ".npy"
                )
            )

            # plt.errorbar(x=az, y=np.mean(thresh_delays), yerr=0.5 * np.std(thresh_delays), fmt='ro', ecolor='r',
            #              capsize=3)
            # plt.errorbar(x=az, y=np.mean(cross_corr_delays), yerr=np.std(cross_corr_delays), fmt='go', ecolor='g',
            #             capsize=3, label='Cross correlation (full signal)')

            plt.errorbar(
                x=az,
                y=np.mean(cross_corr_frame_delays),
                yerr=np.std(cross_corr_frame_delays),
                fmt="ro",
                ecolor="r",
                capsize=3,
                label="Cross correlation (frames) - new",
            )

            # plt.errorbar(x=az, y=np.mean(old_cross_corr_frame_delays), yerr=np.std(old_cross_corr_frame_delays), fmt='bo',
            #              ecolor='b',
            #              capsize=3, label='Cross correlation (frames) - old')
            # plt.errorbar(x=az, y=np.mean(semi_cross_corr_frame_delays), yerr=np.std(semi_cross_corr_frame_delays),
            #              fmt='go',
            #              ecolor='g',
            #              capsize=3, label='Cross correlation (frames) - semi old')

            plt.errorbar(
                x=az,
                y=np.mean(our_new_delays),
                yerr=np.std(our_new_delays),
                fmt="ko",
                ecolor="k",
                capsize=3,
                label="Our",
            )


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
        -5,
        -10,
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
        65,
        80,
    ]  # , 65, 80]

    p = InfoLog(azimuths[0])
    # p.visualize_unique_delays(azimuths, 'GRID')
    p.visualize_unique_delays(azimuths)
    p.visualize_other_methods_delays(azimuths)
    # plt.legend()
    plt.show()
