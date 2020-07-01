import numpy as np
import math


class OnsetModels:
    """
    Class that implements the LSSM with the specified frequencies for all of the onset models. For now, the supported
    models are the decaying sinusoid (with LSSM specified in the method decaying_sinusoid_lssm) and gammatone (with
    LSSM specified in the method gammatone_lssm)
    """

    def __init__(self, f, decay, n_freq):
        # frequency
        self.f = f
        # number of frequencies
        self.n_freq = n_freq
        # decay
        self.decay = decay

    def decaying_sinusoid_lssm(self, n_states):
        """
        Method that contains the LSSM for the decaying sinusoid
        """
        # sampling frequency
        Fs = 44100.0

        self.s = np.repeat(np.array([[1], [0]]), self.n_freq, axis=0)

        A = np.zeros((n_states, n_states))
        for fi in range(self.n_freq):
            # current frequency and decay
            omega = 2 * math.pi * self.f[fi] / Fs
            decay = self.decay[fi]

            # A_p matrix as a block diagonal matrix with rotation matrices at frequency omega
            A[2 * fi : 2 * fi + 2, 2 * fi : 2 * fi + 2] = decay * np.array(
                [[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]
            )
        self.A = A.reshape(n_states, n_states)

    def gammatone_lssm(self, n_states):
        """
        Method that contains the LSSM for the decaying sinusoid
        """
        # sampling frequency
        Fs = 44100.0
        # state of the decaying sinusoid part
        s1 = np.array([[1], [0]])
        # state of the polynomial part
        s2 = np.array([[0], [1], [6], [6]])

        s = np.kron(s1, s2)

        # resulting state vector of the gammatone filter
        self.s = np.repeat(s, self.n_freq, axis=0)

        # state transition matrix for the polynomial part
        A2 = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

        A = np.zeros((n_states, n_states))
        for fi in range(self.n_freq):
            # current frequency and decay
            omega = 2 * math.pi * self.f[fi] / Fs
            decay = self.decay[fi]

            # decaying sinusoid part for the current frequency
            A1 = decay * np.array(
                [[np.cos(omega), -np.sin(omega)], [np.sin(omega), np.cos(omega)]]
            )

            # gammatone filter transition matrix for the current frequency
            A_gammatone = np.kron(A1, A2)

            # stacking the current frequency in the diagonal
            A[8 * fi : 8 * fi + 8, 8 * fi : 8 * fi + 8] = A_gammatone

        self.A = A
