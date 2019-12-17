import numpy as np

from parameters import *


class WindowModels:
    """
    Class that implements the LSSM for all of the window models. For now, the supported windows are the gamma window
    (specified by the method gamma_window_lssm) and the exponential window (specified by the method
    exponential_window_lssm)
    """

    def __init__(self, n_window, decay):
        # state vector
        self.s = np.zeros((n_window, 1))
        # state transition matrix
        self.A = np.zeros((n_window, n_window))
        # output matrix
        self.C = np.zeros((1, n_window))
        # decay
        self.decay = decay

    def gamma_window_lssm(self):
        """
        Method that contains the LSSM for the Gamma window
        """
        A = np.array([[1, 1, 0, 0],
                      [0, 1, 1, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]])
        self.A = self.decay * A
        T_d_1 = np.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 2, 1, 0],
                          [6, 6, 1, 0]])
        C = np.array([0, 0, 0, np.power(1 / Fs, 3)])
        C = np.dot(C, T_d_1)
        self.C = C.reshape(1, 4)
        self.s = np.array([[0],
                           [0],
                           [0],
                           [1]])

    def exponential_window_lssm(self):
        """
        Method that contains the LSSM for the exponential window
        """
        self.A = np.array([self.decay])
        self.s = np.ones(1)
        self.C = np.ones(1)
