import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    audio_data = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/Male_Az_90.csv', delimiter=',')
    y_L = audio_data[:, 0]  # signal from the microphone in the left

    # decaying sinusoid + gamma window
    #lcr = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_0.99_gamma_0.996_Az_90.csv')
    #L = lcr.shape[0]
    # visualizing the results
    #fig, axs = plt.subplots(2)
    #fig.suptitle('Decaying sinusoid + Gamma window')
    #axs[0].plot(range(L), y_L)
    #axs[1].plot(range(L), lcr)
    #plt.show()
    # decaying sinusoid + exponential window
    #lcr = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_0.99_exponential_0.996_Az_90.csv')
    #L = lcr.shape[0]
    # visualizing the results
    #fig, axs = plt.subplots(2)
    #fig.suptitle('Decaying sinusoid + Exponential window')
    #axs[0].plot(range(L), y_L)
    #axs[1].plot(range(L), lcr)
    #plt.show()

    # gammatone + gamma window
    lcr = np.genfromtxt('/Users/gustavocidornelas/Desktop/sound-source/gammatone_0.99_exponential_0.996_Az_90.csv')
    L = lcr.shape[0]
    # visualizing the results
    fig, axs = plt.subplots(2)
    fig.suptitle('Gammatone + Exponential window')
    axs[0].plot(range(L), y_L)
    axs[1].plot(range(L), lcr)
    plt.show()

    # gammatone + exponential window