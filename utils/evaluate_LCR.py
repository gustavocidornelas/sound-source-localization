import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from matplotlib import rc

if __name__ == '__main__':
    # audio data
    audio = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/ISTS Signal/Speech signals/Speech_Az_-80.csv', delimiter=',')
    audio = audio[1000:, :]

    # read cs file with the LCR
    LCR_1 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_[0.99]_gamma_0.999_Az_-55_freq_[80.0].csv', delimiter=",", skip_header=True)
    LCR_1 = LCR_1[1000:, :]

    #transform_1 = abs(np.fft.fftshift(np.fft.fft(LCR_1[:, 0])))
    #BW_1 = abs(np.argmax(transform_1) - np.argmax(transform_1 > 1e-3))

    #LCR_2 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/analysis/gammatone_0.987_gamma_0.999_Az_'
    #                      '90_freq_80.0.csv', delimiter=",", skip_header=True)
    #LCR_2 = LCR_2[30000:, :]
    #transform_2 = abs(np.fft.fftshift(np.fft.fft(LCR_2[:, 0])))
    #BW_2 = abs(np.argmax(transform_2) - np.argmax(transform_2 > 1e-3))

    #LCR_3 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/analysis/decaying_sinusoid_0.9_gamma_0.997_Az_'
    #                      '90_freq_80.0.csv', delimiter=",", skip_header=True)
    #LCR_3 = LCR_3[500:, :]
    #transform_3 = abs(np.fft.fftshift(np.fft.fft(LCR_3[:, 0])))
    #BW_3 = abs(np.argmax(transform_3) - np.argmax(transform_3 > 1e-3))

    #LCR_4 = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/analysis/decaying_sinusoid_0.99_gamma_0.999_Az_'
    #                      '90_freq_80.0.csv', delimiter=",", skip_header=True)
    #LCR_4 = LCR_4[500:, :]
    #transform_4 = abs(np.fft.fftshift(np.fft.fft(LCR_4[:, 0])))
    #BW_4 = abs(np.argmax(transform_4) - np.argmax(transform_4 > 1e-3))

    # calculating the entropy for the LCRs
    #entropy_1 = entropy(LCR_1 / sum(LCR_1))
    #entropy_2 = entropy(LCR_2 / sum(LCR_2))
    #entropy_3 = entropy(LCR_3 / sum(LCR_3))
    #entropy_4 = entropy(LCR_4 / sum(LCR_4))
    #print(entropy_1)
    #print(entropy_2)
    #print(entropy_3)
    #print(entropy_4)

    rc('text', usetex=True)
    fig, axs = plt.subplots(2)
    axs[0].set_title('Audio signal')
    axs[0].plot(np.asarray(range(audio.shape[0])) / 44100.0, audio[:, 0], 'b')
    axs[0].plot(np.asarray(range(audio.shape[0])) / 44100.0, audio[:, 1], 'r')
    axs[1].plot(np.asarray(range(LCR_1.shape[0])) / 44100.0, LCR_1[:, 1], 'b', linewidth=1.2)
    axs[1].plot(np.asarray(range(LCR_1.shape[0])) / 44100.0, LCR_1[:, 2], 'r', linewidth=1.2)
    #axs[1].plot(np.asarray(range(LCR_1.shape[0])) / 44100.0, LCR_1[:, 2], 'b', linewidth=1.2)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_title('Linear combination')
    axs[0].grid(ls='--', c='.5')
    axs[1].grid(ls='--', c='.5')


    plt.figure()
    plt.plot(np.asarray(range(LCR_1.shape[0])), LCR_1[:, 1], 'b', linewidth=1.2)
    plt.plot(np.asarray(range(LCR_1.shape[0])), LCR_1[:, 2], 'r', linewidth=1.2)
    plt.grid(ls='--', c='.5')
    plt.title('f = 80 Hz')
    plt.xlabel('k')
    #plt.figure()
    #plt.plot(range(LCR_2.shape[0]), LCR_2[:, 1])
    #plt.plot(range(LCR_2.shape[0]), LCR_2[:, 2])

    plt.show()

    #audio_data = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/Male_Az_90.csv', delimiter=',')
    #y_L = audio_data[500:, 0]  # signal from the microphone in the left
    #y_R = audio_data[500:, 2]  # signal from the microphone in the right

    # get rid of the first 500 samples
    #LCR = LCR[500:, :]

    # calculating the Fourier transform
    #transform = np.fft.fftshift(np.fft.fft(LCR[:, 0]))
    #magn = abs(transform)
    #plt.semilogy(range(magn.shape[0]), magn)
    #plt.show()

    #fig, axs = plt.subplots(4)
    #fig.suptitle('[Decaying sinusoid - gamma window]')
    #L = LCR_1.shape[0]
    # axs[0].semilogy(range(transform_1.shape[0]), transform_1, label='Decaying sinusoid + gamma: 0n .99, W .999')
    #axs[0].legend(loc="upper right")
    #axs[1].semilogy(range(transform_2.shape[0]), transform_2, label='Gammatone + gamma: 0n .987, W .999')
    #axs[1].legend(loc="upper right")
    #axs[2].semilogy(range(transform_3.shape[0]), transform_3, label='0n .99, W .999')
    #axs[2].legend(loc="upper right")
    #axs[3].semilogy(range(transform_4.shape[0]), transform_4, label='On .99, W .998')
    #axs[3].legend(loc="upper right")
    #plt.show()

    #fig, axs = plt.subplots(4)
   # fig.suptitle('Best models - gamma window')
    #L = LCR_1.shape[0]
    #axs[0].plot(range(transform_1.shape[0]), LCR_1[:, 1], label='Decaying sinusoid + gamma: 0n .99, W .999')
    #axs[0].plot(range(transform_1.shape[0]), LCR_1[:, 2])
    #axs[0].legend(loc="upper right")
    #axs[1].plot(range(transform_2.shape[0]), LCR_2[:, 1], label='Gammatone + gamma: 0n .987, W .999')
    #axs[1].plot(range(transform_2.shape[0]), LCR_2[:, 2])
    #axs[1].legend(loc="upper right")
    #axs[2].plot(range(transform_3.shape[0]), LCR_3[:, 1], label='Decaying sinusoid + gamma: 0n .9, W .997')
    #axs[2].plot(range(transform_3.shape[0]), LCR_3[:, 2])
    #axs[2].legend(loc="upper right")
    #axs[3].plot(range(transform_4.shape[0]), LCR_4[:, 1], label='On .99, W .996')
    #axs[3].plot(range(transform_4.shape[0]), LCR_4[:, 2], label='On .99, W .996')
    #axs[3].legend(loc="upper right")
    #plt.show()

    #plt.scatter(range(4), np.array([BW_1, BW_2, BW_3, BW_4]))
    #plt.show()