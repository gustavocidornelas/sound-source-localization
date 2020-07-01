import numpy as np
import matplotlib.pyplot as plt

from compute_LCR_online import *
from poly_fit_online import *
from delay_estimation_online import *

if __name__ == "__main__":
    # simulation parameters
    audio_file = "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_45.csv"
    frequencies = [80.0]
    n_freq = 1
    onset_decays = [0.99]
    window_decay = [0.999]
    onset_model = "decaying_sinusoid"
    window_model = "gamma"
    burn_in = (
        30000  # number of samples of the the LCR that are thrown away at the beginning
    )

    # reading the audio file
    audio = np.genfromtxt(audio_file, delimiter=",", skip_header=False)
    # audio = audio[:150200, :]

    # creating the LCR module and initializing the messages
    LCR_mod = ComputeLCROnline(
        onset_model, window_model, frequencies, n_freq, onset_decays, window_decay
    )
    chi_k = np.zeros((LCR_mod.n_window, 1))
    zeta_k = np.zeros((LCR_mod.n_window * LCR_mod.n_states, audio.shape[1]))
    s_k = np.zeros((LCR_mod.n_window * LCR_mod.n_states, LCR_mod.n_states))
    w_k = np.zeros((LCR_mod.n_states, LCR_mod.n_states))
    xi_k = np.zeros((LCR_mod.n_states, audio.shape[1]))
    k_k = 0

    # creating the polynomial fitting module and initializing the messages
    poly_mod = FitPolyOnline()
    k_k_f_L = 0
    k_k_b_L = 0
    xi_k_f_L = np.zeros((poly_mod.n_states, 1))
    xi_k_b_L = np.zeros((poly_mod.n_states, 1))
    W_k_f_L = np.zeros((poly_mod.n_states, poly_mod.n_states))
    W_k_b_L = np.zeros((poly_mod.n_states, poly_mod.n_states))

    k_k_f_R = 0
    k_k_b_R = 0
    xi_k_f_R = np.zeros((poly_mod.n_states, 1))
    xi_k_b_R = np.zeros((poly_mod.n_states, 1))
    W_k_f_R = np.zeros((poly_mod.n_states, poly_mod.n_states))
    W_k_b_R = np.zeros((poly_mod.n_states, poly_mod.n_states))

    # creating the delay estimation module
    delay_mod = DelayEstimationOnline()

    # array to store the LCR
    LCR = np.zeros((audio.shape[0], audio.shape[1]))
    LCR_window_L = np.zeros(201)
    LCR_window_R = np.zeros(201)

    # array to store the polynomial coefficients
    C_opt_L = np.zeros((audio.shape[0], 4))
    C_opt_R = np.zeros((audio.shape[0], 4))

    # array to store all the delay estimates
    total_delays = np.zeros(audio.shape[0])
    count = 0

    # figure that continuously displays the results
    fig_window = int(0.5 * 44100)
    chunk = 1024
    y_left = np.zeros(fig_window)
    y_right = np.zeros(fig_window)
    lcr_left = np.zeros(fig_window)
    lcr_right = np.zeros(fig_window)
    delay_time = np.zeros(fig_window)
    time = np.zeros(fig_window)
    time_chunk = []
    y_left_chunk = []
    y_right_chunk = []
    lcr_left_chunk = []
    lcr_right_chunk = []
    delay_time_chunk = []
    fig, axs = plt.subplots(3)
    axs[0].set_ylabel("Audio signal")
    axs[0].set_ylim(-0.15, 0.15)
    axs[1].set_ylabel("LCRs")
    axs[1].set_ylim(0, 0.0003)
    axs[2].set_ylabel("Delay [ms]")
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel("Time [s]")
    axs[0].grid(ls="--", c=".5")
    axs[1].grid(ls="--", c=".5")
    axs[2].grid(ls="--", c=".5")

    delay = 0
    # loop that goes through the audio signal
    for k in range(audio.shape[0]):

        # current audio sample
        y = audio[k, :]

        # computing the LCR for the audio sample
        chi_k, zeta_k, s_k, w_k, xi_k, k_k, LCR[k, :] = LCR_mod.compute_lcr(
            y, chi_k, zeta_k, s_k, w_k, xi_k, k_k
        )

        # buffer
        # filling the buffer with LCR samples
        if burn_in < k <= burn_in + 201:
            LCR_window_L[k - burn_in - 1] = LCR[k, 0]
            LCR_window_R[k - burn_in - 1] = LCR[k, 1]
        if k == burn_in + 201:
            # initialize the messages placing the window in the first position
            (
                k_k_f_L,
                k_k_b_L,
                xi_k_f_L,
                xi_k_b_L,
                W_k_f_L,
                W_k_b_L,
                C_opt_L[k, :],
                LCR_window_L,
            ) = poly_mod.initialize_messages(
                LCR_window_L, k_k_f_L, k_k_b_L, xi_k_f_L, xi_k_b_L, W_k_f_L, W_k_b_L
            )
            (
                k_k_f_R,
                k_k_b_R,
                xi_k_f_R,
                xi_k_b_R,
                W_k_f_R,
                W_k_b_R,
                C_opt_R[k, :],
                LCR_window_R,
            ) = poly_mod.initialize_messages(
                LCR_window_R, k_k_f_R, k_k_b_R, xi_k_f_R, xi_k_b_R, W_k_f_R, W_k_b_R
            )
        elif k > burn_in + 201:
            # moving the window forward
            LCR_new_L = LCR[k, 0]
            (
                k_k_f_L,
                k_k_b_L,
                xi_k_f_L,
                xi_k_b_L,
                W_k_f_L,
                W_k_b_L,
                C_opt_L[k, :],
                LCR_window_L,
            ) = poly_mod.fit_polynomial(
                LCR_window_L,
                LCR_new_L,
                k_k_f_L,
                k_k_b_L,
                xi_k_f_L,
                xi_k_b_L,
                W_k_f_L,
                W_k_b_L,
            )
            LCR_new_R = LCR[k, 1]
            (
                k_k_f_R,
                k_k_b_R,
                xi_k_f_R,
                xi_k_b_R,
                W_k_f_R,
                W_k_b_R,
                C_opt_R[k, :],
                LCR_window_R,
            ) = poly_mod.fit_polynomial(
                LCR_window_R,
                LCR_new_R,
                k_k_f_R,
                k_k_b_R,
                xi_k_f_R,
                xi_k_b_R,
                W_k_f_R,
                W_k_b_R,
            )
            # estimating the delay
            count = count + 1
            total_delays[k], count = delay_mod.estimate_delay(
                coeff_left_k=C_opt_L[k, :],
                coeff_right_k=C_opt_R[k, :],
                count=count,
                delay=total_delays[k - 1],
            )

        if k != 0:
            time_chunk.append(k)
            y_left_chunk.append(y[0])
            y_right_chunk.append(y[1])
            lcr_left_chunk.append(LCR[k, 0])
            lcr_right_chunk.append(LCR[k, 1])
            delay_time_chunk.append(total_delays[k])

        # preparing the plots
        if k % chunk == 0 and k != 0:
            time[:-chunk] = time[chunk:]
            time[-chunk:] = time_chunk
            y_left[:-chunk] = y_left[chunk:]
            y_left[-chunk:] = y_left_chunk
            y_right[:-chunk] = y_right[chunk:]
            y_right[-chunk:] = y_right_chunk
            lcr_left[:-chunk] = lcr_left[chunk:]
            lcr_left[-chunk:] = lcr_left_chunk
            lcr_right[:-chunk] = lcr_right[chunk:]
            lcr_right[-chunk:] = lcr_right_chunk
            delay_time[:-chunk] = delay_time[chunk:]
            delay_time[-chunk:] = delay_time_chunk

            axs[0].set_xlim((max(time) - fig_window) / 44100, max(time) / 44100)
            axs[0].plot(time[::32] / 44100, y_right[::32], "b", label="Right")
            axs[0].plot(time[::32] / 44100, y_left[::32], "r", label="Left")
            if k == chunk:
                axs[0].legend(loc="upper right")

            axs[1].set_xlim((max(time) - fig_window) / 44100, max(time) / 44100)
            axs[1].plot(time[::32] / 44100, lcr_left[::32], "r")
            axs[1].plot(time[::32] / 44100, lcr_right[::32], "b")
            axs[2].set_xlim((max(time) - fig_window) / 44100, max(time) / 44100)
            axs[2].plot(time[::32] / 44100, delay_time[::32] / 44.1, "k")

            time_chunk = []
            y_left_chunk = []
            y_right_chunk = []
            lcr_left_chunk = []
            lcr_right_chunk = []
            delay_time_chunk = []

            plt.pause(0.05)
