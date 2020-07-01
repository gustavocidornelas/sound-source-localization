import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize
from scipy import stats

from matplotlib import rc


if __name__ == "__main__":

    def test_func(x, a, b, c, d):
        return a * np.power(x, 3) + b * np.power(x, 2) + c * x + d

    median_delay = []
    theo_delay = []
    mean_delay = []
    mode_delay = []
    teste = []

    x_data = np.zeros(106)
    y_data = np.zeros(106)

    data_list = []

    azimuths = [
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

    points = np.zeros(25 * 3)

    for az in azimuths:
        # reading the file with the delays
        delays = np.genfromtxt(
            "/Users/gustavocidornelas/Desktop/sound-source/unique_delays_Az_"
            + str(az)
            + "_freq_[80.0].csv",
            delimiter=",",
            skip_header=False,
        )

        # delays = np.load('/Users/gustavocidornelas/Desktop/sound-source/teste_all_delays_Az_' + str(az) + '.npy')

        delays = delays[1:] / 44.1
        median_delay.append(np.median(delays))
        mean_delay.append(np.mean(delays))
        mode_delay.append(stats.mode(delays)[0])

        teste.append((delays[1] + delays[2]) / 2)
        delays = delays[2:4]

        # if az < 0 and az >= -90:m
        #    delays = delays * -1
        # if az > 90:
        #    delays = delays * -1
        # if az < 0:
        #    delays = delays * -1

        # num_elements = delays.shape[0]

        # storing x and y values
        # plt.scatter(np.repeat(az, delays.shape[0]), delays)
        data_list.append(np.vstack((np.repeat(az, delays.shape[0]), delays)))

        theo_delay.append(((152 / 343) * np.sin(np.deg2rad(az))))

        # sns.violinplot(x=np.repeat(az, delays.shape[0]), y=delays)
        # azimuth_list.append(az)
        # delay_list.append(np.median(delays))

    dataset = np.concatenate(data_list, axis=1)
    x_data = dataset[0, :]
    y_data = dataset[1, :]

    # x_data = x_data[29:81]
    # y_data = y_data[29:81]

    # mapping to the +- 90 degrees interval
    # x_data[:29] = x_data[:29] + 180
    # x_data[82:] = x_data[82:] - 180
    params, _ = optimize.curve_fit(test_func, x_data, y_data)

    y_true = test_func(x_data, params[0], params[1], params[2], params[3])

    MSE = sum((y_data - y_true) ** 2) / y_data.shape[0]

    print("The MSE is " + str(MSE))

    # print(params)
    # plt.scatter(x_data, y_data)
    # rc('text', usetex=True)
    # plt.rcParams.update({'font.size': 16})
    # plt.plot(range(-90, 90, 10), test_func(range(-90, 90, 10), params[0], params[1], params[2], params[3]), 'k',
    #         label='Least squares fit', linewidth=2)
    # thresh_delays = np.load('/Users/gustavocidornelas/Desktop/sound-source/Other methods/delays_threshold.npy')
    # plt.plot(range(-180, 180, 10), thresh_delays, 'r')
    # corr_delays = np.load('/Users/gustavocidornelas/Desktop/sound-source/Other methods/delays_cross_corr.npy')
    # plt.plot(range(-180, 180, 10), corr_delays, 'b')
    # plt.plot(range(-180, 180, 10), theo_delay, label="Median delay", linewidth=2)
    # plt.legend()
    # plt.xlabel('Azimuth [degrees]')
    # plt.ylabel('Delay [ms]')
    # plt.grid(ls='--', c='.5')
    # plt.show()

    # loading the delays computed with the alternative methods
    thresh_delays = np.load(
        "/Users/gustavocidornelas/Desktop/sound-source/Other methods/delays_threshold.npy"
    )
    corr_delays = np.load(
        "/Users/gustavocidornelas/Desktop/sound-source/Other methods/delays_cross_corr.npy"
    )
    group_delays = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/group_delays.csv",
        delimiter=",",
    )

    azimuths = [
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

    # creating the plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    # ax.scatter(np.deg2rad(x_data), abs(y_data))
    # ax.plot(np.deg2rad(range(-180, 180, 10)), abs(np.array(mean_delay)), label='Our algorithm (median delay)', linewidth=2)
    ax.plot(
        np.deg2rad(np.array(azimuths)),
        abs(thresh_delays),
        label="Threshold method (-30 dB)",
        linewidth=2,
    )
    # plt.legend()
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='polar')
    # ax.set_theta_zero_location("N")
    ax.scatter(np.deg2rad(x_data), y_data, color="red", label="Our algorithm")
    # ax.plot(np.deg2rad(azimuths), teste, label='Our algorithm', color='red', linewidth=2)
    # ax.plot(np.deg2rad(azimuths), teste)
    # ax.scatter(x_data, y_data)
    ax.plot(
        np.deg2rad(np.array(azimuths)),
        abs(np.array(theo_delay)),
        label="Theoretical delay",
        linewidth=2,
    )
    ax.plot(
        np.deg2rad(np.array(azimuths)),
        abs(corr_delays),
        label="Max cross-correlation method",
        linewidth=2,
    )
    ax.plot(
        np.deg2rad(np.array(azimuths)),
        abs(group_delays * 1e4),
        label="Group delay (multiplied by 1e4)",
        linewidth=2,
    )
    plt.legend()
    plt.show()
