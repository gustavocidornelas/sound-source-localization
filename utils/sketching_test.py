import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy import signal
import sounddevice as sd


if __name__ == "__main__":

    # # loading the files to play with
    # file_directory = '/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_-55'
    # lcr = np.genfromtxt(file_directory + '/LCR/f_sig_s4_decaying_sinusoid_[0.99]_gamma_0.999_Az_-55_freq_[240.0].csv',
    #                     delimiter=',', skip_header=True)
    # lcr_left = lcr[30000:, 1]
    # lcr_right = lcr[30000:, 2]
    # coeff_left = np.genfromtxt(file_directory + '/Coefficients/f_sig_s4_coeff_left_Az_-55_freq_[200.0].csv',
    #                            delimiter=',', skip_header=False)
    # coeff_right = np.genfromtxt(file_directory + '/Coefficients/f_sig_s4_coeff_right_Az_-55_freq_[200.0].csv',
    #                             delimiter=',', skip_header=False)

    # investigating what's going on with the stereo signals
    # HRIRs saved on my computer (used in the convolutions to generate the dataset)
    hrir_saved = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/HRIR/@44.1kHz/HRIR_Az_-80.csv",
        delimiter=",",
    )
    hrir_saved_left = hrir_saved[:, 0]
    hrir_saved_right = hrir_saved[:, 1]

    # original HRIRs (CIPIC)
    fs, hrir_original_left = wavfile.read(
        "/Users/gustavocidornelas/Downloads/wav_database/subject165/neg80azleft.wav"
    )
    fs, hrir_original_right = wavfile.read(
        "/Users/gustavocidornelas/Downloads/wav_database/subject165/neg80azright.wav"
    )
    hrir_original_left = hrir_original_left[8, :]
    hrir_original_right = hrir_original_right[8, :]

    # using the ISTS signals
    ists_signal_stereo = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech_"
        "Az_-80.csv",
        delimiter=",",
    )
    plt.figure()
    plt.plot(range(ists_signal_stereo.shape[0]), ists_signal_stereo[:, 0])
    plt.title("matlab")

    # reading the chosen signal
    fs, ists_signal_mono = wavfile.read(
        "/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/NFIMenglish.wav"
    )

    # trying to get to the same result as ists_signal_stereo
    left_ists = np.convolve(hrir_original_left, ists_signal_mono, mode="same")
    right_ists = np.convolve(hrir_original_right, ists_signal_mono, mode="same")
    rep_ists_stereo = np.transpose(np.vstack((left_ists, right_ists)))

    # using f_sig_s4
    fs, sig = wavfile.read(
        "/Users/gustavocidornelas/Desktop/sound-source/DAta GRID/Female/s4/prai8s.wav"
    )
    delay_signal = np.load(
        "/Users/gustavocidornelas/Desktop/sound-source/Dataset from GRID/Az_-80/f_sig_s4_Az_-80.npy"
    )

    # convolving the impulse responses with the speech signal
    speech_left = signal.convolve(sig, hrir_saved_left, mode="same")
    speech_right = signal.convolve(sig, hrir_original_right)

    # multi-channel signal
    # speech_carai = np.transpose(np.vstack((speech_left, speech_right)))

    plt.figure()
    plt.plot(range(delay_signal.shape[0]), delay_signal[:, 0])
    plt.title("python")
    plt.show()

    # cropping the region of interest of the LCR
    # roi_low = 26000 #32800
    # roi_high =28000 # 35000
    # lcr_left = lcr_left[roi_low:roi_high]
    # lcr_right = lcr_right[roi_low:roi_high]
    # coeff_left = coeff_left[roi_low:roi_high, :]
    # coeff_right = coeff_right[roi_low:roi_high, :]
    #
    #
    # plt.figure()
    # plt.plot(range(roi_low, roi_high), lcr_left, 'b')
    # plt.plot(range(roi_low, roi_high), lcr_right, 'r')
    # #plt.xlim([25600, 26000])
    # plt.show()
    # #plt.scatter(33149, lcr_left[33149-roi_low], marker='x', s=100, color='b')
    # #plt.scatter(33149, lcr_right[33149 - roi_low], marker='x', s=100, color='r')
    # #plt.axvline(x=33149 - 100, color='k', linestyle='--')
    # #plt.axvline(x=33149 + 100, color='k', linestyle='--', label='3rd degree polynomial window')
    # x = range(-100, 100)
    #
    # coeff_left_k = coeff_left[33149 - roi_low, :]
    # coeff_right_k = coeff_right[33149 - roi_low, :]
    #
    # poly_left = coeff_left_k[0] + coeff_left_k[1] * x + coeff_left_k[2] * np.power(x, 2) + coeff_left_k[
    #     3] * np.power(x, 3)
    # poly_right = coeff_right_k[0] + coeff_right_k[1] * x + coeff_right_k[2] * np.power(x, 2) + coeff_right_k[
    #     3] * np.power(x, 3)
    #
    # plt.plot(range(33149 - 100, 33149 + 100), poly_left, 'k--')
    # plt.plot(range(33149 - 100, 33149 + 100), poly_right, 'k--', label='3rd degree polynomial')
    # #plt.xlim([25500, 26500])
    #
    # # visualizing the sketch
    # sketch_interval = 1
    # num_sketches = 15 #int(coeff_left.shape[0] / sketch_interval)
    #
    # #for k in range(num_sketches):
    # #    coeff_left_k = coeff_left[33149 + k * sketch_interval - roi_low, :]
    # #    coeff_right_k = coeff_right[33149 + k * sketch_interval - roi_low, :]
    #
    # #    poly_left = coeff_left_k[0] + coeff_left_k[1] * x + coeff_left_k[2] * np.power(x, 2) + coeff_left_k[
    # #        3] * np.power(x, 3)
    # #    poly_right = coeff_right_k[0] + coeff_right_k[1] * x + coeff_right_k[2] * np.power(x, 2) + coeff_right_k[
    # #        3] * np.power(x, 3)
    # #    plt.plot(range(33149 + k * sketch_interval - 100, 33149 + k * sketch_interval + 100), poly_left, 'k--')
    # #    plt.plot(range(33149 + k * sketch_interval - 100, 33149 + k * sketch_interval + 100), poly_right, 'k--')
    #
    #
    # plt.ylim([0, 0.0018])
    # #plt.xlim([roi_low, roi_high])
    # #plt.legend()
    #
    # # borders of the new window
    # plt.axvline(x=33149 - 40, color='g', linestyle='--')
    # plt.axvline(x=33149 + 40, color='g', linestyle='--', label='2nd degree polynomial window')
    # b1 = -40
    # b2 = 40
    #
    # plt.scatter(33149 + b1, coeff_left_k[0] + coeff_left_k[1] * b1 + coeff_left_k[2] * np.power(b1, 2) + coeff_left_k[
    #     3] * np.power(b1, 3), marker='o', color='g')
    # plt.scatter(33149 + b2, coeff_left_k[0] + coeff_left_k[1] * b2 + coeff_left_k[2] * np.power(b2, 2) + coeff_left_k[
    #     3] * np.power(b2, 3), marker='o')
    # plt.scatter(33149 + b1, coeff_right_k[0] + coeff_right_k[1] * b1 + coeff_right_k[2] * np.power(b1, 2) + coeff_right_k[
    #     3] * np.power(b1, 3), marker='o', color='g')
    # plt.scatter(33149 + b2, coeff_right_k[0] + coeff_right_k[1] * b2 + coeff_right_k[2] * np.power(b2, 2) + coeff_right_k[
    #     3] * np.power(b2, 3), marker='o')
    #
    # # generating 2nd degree approximation
    # a0 = -40.0
    # b0 = 40.0
    # A = np.array([[1, 0, 0],
    #               [1, a0, np.power(a0, 2)],
    #               [1, b0, np.power(b0, 2)]])
    # A_inv = np.linalg.inv(A)
    #
    # beta_left = np.dot(A_inv, np.array([[coeff_left_k[0]],
    #                                     [coeff_left_k[0] + coeff_left_k[1] * a0 +
    #                                      coeff_left_k[2] * np.power(a0, 2) + coeff_left_k[3]
    #                                      * np.power(a0, 3)],
    #                                     [coeff_left_k[0] + coeff_left_k[1] * b0 +
    #                                      coeff_left_k[2] * np.power(b0, 2) + coeff_left_k[3]
    #                                      * np.power(b0, 3)]
    #                                     ]))
    # beta_right = np.dot(A_inv, np.array([[coeff_right_k[0]],
    #                                      [coeff_right_k[0] + coeff_right_k[1] * a0 +
    #                                       coeff_right_k[2] * np.power(a0, 2) + coeff_right_k[3]
    #                                       * np.power(a0, 3)],
    #                                      [coeff_right_k[0] + coeff_right_k[1] * b0 +
    #                                       coeff_right_k[2] * np.power(b0, 2) + coeff_right_k[3]
    #                                       * np.power(b0, 3)]
    #                                      ]))
    #
    # # evaluating the 2nd degree polynomial inside the window
    # t = range(-40, 40)
    # poly_2_left = beta_left[0] + beta_left[1] * t + beta_left[2] * np.power(t, 2)
    # poly_2_right = beta_right[0] + beta_right[1] * t + beta_right[2] * np.power(t, 2)
    #
    # plt.plot(range(33149-40, 33149+40), poly_2_left, 'b', linewidth=2)
    # plt.plot(range(33149 - 40, 33149 + 40), poly_2_right, 'r', linewidth=2, label='2nd degree polynomial')
    #
    #
    # # estimating the delay
    # plt.axhline(y=beta_right[0], color='g', linestyle='--')
    #
    # plt.scatter(33149, beta_right[0], marker='d', s=200, color='k')
    #
    # roots = np.roots(np.array([beta_left[2, 0], beta_left[1, 0], beta_left[0, 0] - beta_right[0, 0]]))    # estimating the delay based on the roots
    # if np.isreal(roots).all():
    #     # delay = get_delay(roots)
    #     if roots[0] > 0 and roots[1] > 0:
    #         delay = np.min(roots)
    #     elif roots[0] < 0 and roots[1] > 0:
    #         delay = roots[1]
    #     elif roots[0] > 0 and roots[1] < 0:
    #         delay = roots[0]
    #
    # plt.scatter(33149 + delay, beta_right[0], marker='d', s=200, color='k')
    # plt.axhline(y=beta_right[0], xmin=33149 / roi_high, xmax=(33149 + delay)/ roi_high, color='k')
    #
    #
    #
    # #plt.xlim([25845-40, 25845+40])
    #
    # plt.legend()
    #
    #
    #
    # plt.show()
    # extracting the signature
    # sig_left = 190
    # sig_right = 140
    # coeff_left_k = coeff_left[sig_left, :]
    # coeff_right_k = coeff_right[sig_right, :]
    # poly_left = coeff_left_k[0] + coeff_left_k[1] * x + coeff_left_k[2] * np.power(x, 2) + coeff_left_k[
    #     3] * np.power(x, 3)
    # poly_right = coeff_right_k[0] + coeff_right_k[1] * x + coeff_right_k[2] * np.power(x, 2) + coeff_right_k[
    #     3] * np.power(x, 3)
    #
    # #plt.plot(range(sig_left - 100, sig_left + 100), poly_left, 'y', linewidth=2)
    # #plt.plot(range(sig_right - 100, sig_right + 100), poly_right, 'y', linewidth=2)
    #
    # plt.figure()
    # x = range(-300, 300)
    # poly_left = coeff_left_k[0] + coeff_left_k[1] * x + coeff_left_k[2] * np.power(x, 2) + coeff_left_k[
    #     3] * np.power(x, 3)
    # poly_right = coeff_right_k[0] + coeff_right_k[1] * x + coeff_right_k[2] * np.power(x, 2) + coeff_right_k[
    #     3] * np.power(x, 3)
    # plt.plot(range(sig_left - 300, sig_left + 300), poly_left, 'b', linewidth=2)
    # plt.plot(range(sig_right - 300, sig_right + 300), poly_right, 'r', linewidth=2)
    # plt.axvline(x=sig_left-100, color='b', linestyle='--')
    # plt.axvline(x=sig_left+100, color='b', linestyle='--')
    # plt.axvline(x=sig_right - 100, color='r', linestyle='--')
    # plt.axvline(x=sig_right + 100, color='r', linestyle='--')
    # plt.show()
