import numpy as np
import math

from parameters import *


class OnsetModels:
    """
    Class that implements the LSSM with the specified frequencies for all of the onset models. For now, the supported
    models are the decaying sinusoid (with LSSM specified in the method decaying_sinusoid_lssm) and gammatone (with
    LSSM specified in the method gammatone_lssm)
    """
    def __init__(self, f, decay):
        # frequency
        self.f = f
        # state vector
        self.s = np.zeros((n, n_freq))
        # state transition matrix
        self.A = np.zeros((n_states, n_states))
        # decay
        self.decay = decay

    def decaying_sinusoid_lssm(self):
        """
        Method that contains the LSSM for the decaying sinusoid
        """
        self.s = np.repeat(np.array([[1],
                                     [0]]), n_freq, axis=0)

        A = np.zeros((n_states, n_states))
        for fi in range(n_freq):
            # current frequency in the frequency bank
            omega = 2 * math.pi * self.f[fi] / Fs
            # A_p matrix as a block diagonal matrix with rotation matrices at frequency omega
            A[2 * fi:2 * fi + 2, 2 * fi:2 * fi + 2] = np.array([[np.cos(omega), -np.sin(omega)],
                                                                [np.sin(omega), np.cos(omega)]])
        A = self.decay * A
        self.A = A.reshape(n_states, n_states)

    def gammatone_lssm(self):
        """
        Method that contains the LSSM for the decaying sinusoid
        """
        # state for the decaying sinusoid part
        s1 = np.repeat(np.array([[1],
                                 [0]]), n_freq, axis=0)

        # state for the polynomial part
        s2 = np.array([[0],
                       [1],
                       [6],
                       [6]])

        # state for the gammatone
        self.s = np.kron(s1, s2)

        # state transition matrix for the decaying sinusoid part ######### FIX this later
        A1 = np.zeros((2, 2))
        for fi in range(n_freq):
            # current frequency in the frequency bank
            omega = 2 * math.pi * self.f[fi] / Fs
            # A_p matrix as a block diagonal matrix with rotation matrices at frequency omega
            A1[2 * fi:2 * fi + 2, 2 * fi:2 * fi + 2] = np.array([[np.cos(omega), -np.sin(omega)],
                                                                 [np.sin(omega), np.cos(omega)]])
        A1 = self.decay * A1
        A1 = A1.reshape(2, 2)

        # state transition matrix for the polynomial part
        A2 = np.array([[1, 1, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 1],
                       [0, 0, 0, 1]])

        # state transition matrix for the gammatone
        self.A = np.kron(A1, A2)

