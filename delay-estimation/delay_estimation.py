"""
Script that estimates the delay between two sound signals based on the local polynomial approximation of their
corresponding LCRs

Author: Gustavo Cid Ornelas, ETH Zurich, November 2019
"""
import numpy as np
import matplotlib.pyplot as plt


def get_delay(roots):
    """
    Function that estimates the delay between the two signals based on the roots found on the second degree-polynomial
    approximation

    Parameter
    -----
    roots (np.array): array with the roots found when estimating the shift between the signals

    Returns
    -----
    delay (float): delay estimate
    """
    if roots[0] > 0 and roots[1] > 0:
        delay = np.min(roots)
    elif roots[0] < 0 and roots[1] > 0:
        delay = roots[1]
    elif roots[0] > 0 and roots[1] < 0:
        delay = roots[0]

    return delay


if __name__ == '__main__':
    # loading the files with the coefficients of the corresponding signals LCRs (in the canonical basis)
    coeff_left = np.genfromtxt('transformed_poly_fit_result_LEFT.csv', delimiter=',', skip_header=False)
    coeff_right = np.genfromtxt('transformed_poly_fit_result_RIGHT.csv', delimiter=',', skip_header=False)

    # some useful computations and parameters
    # window size for the 2nd degree polynomial approximation
    a0 = -40.0
    b0 = 40.0
    A = np.array([[1, 0, 0],
                  [1, a0, np.power(a0, 2)],
                  [1, b0, np.power(b0, 2)]])
    A_inv = np.linalg.inv(A)

    # array that stores the estimated delays
    total_delay = np.zeros(coeff_left.shape[0])
    delay = 0

    # main loop, going through each sample coefficients
    for k in range(coeff_left.shape[0]):
        # retrieving the current sample's coefficients
        coeff_left_k = coeff_left[k, :]
        coeff_right_k = coeff_right[k, :]

        # checking if it is a good sample to estimate the delay (both increasing signals)
        if(coeff_left_k[1] > 1.5e-7 and 2 * coeff_left_k[2] < 1e-13 and
           coeff_right_k[1] > 1.5e-7 and 2 * coeff_right_k[2] < 1e-13):

            # approximating the 3rd degree polynomials by 2nd degree polynomials (coefficients beta) on a smaller window
            beta_left = np.dot(A_inv, np.array([[coeff_left_k[0]],
                                                [coeff_left_k[0] + coeff_left_k[1] * a0 +
                                                 coeff_left_k[2] * np.power(a0, 2) + coeff_left_k[3] * np.power(a0, 3)],
                                                [coeff_left_k[0] + coeff_left_k[1] * b0 +
                                                 coeff_left_k[2] * np.power(b0, 2) + coeff_left_k[3] * np.power(b0, 3)]
                                                ]))
            beta_right = np.dot(A_inv, np.array([[coeff_right_k[0]],
                                                [coeff_right_k[0] + coeff_right_k[1] * a0 +
                                                 coeff_right_k[2] * np.power(a0, 2) + coeff_right_k[3] * np.power(a0, 3)],
                                                [coeff_right_k[0] + coeff_right_k[1] * b0 +
                                                 coeff_right_k[2] * np.power(b0, 2) + coeff_right_k[3] * np.power(b0, 3)]
                                                ]))

            # checking which 2nd degree polynomial is in front
            if beta_left[0, 0] > beta_right[0, 0]:
                # solving the system to find when the signal in the right reaches the same level as the one in the left
                roots = np.roots(np.array([beta_right[2, 0], beta_right[1, 0], beta_right[0, 0] - beta_left[0, 0]]))

                # estimating the delay based on the roots
                if np.isreal(roots).all():
                    delay = get_delay(roots)

            else:
                # solving the system to find when the signal in the left reaches the same level as the one in the right
                roots = np.roots(np.array([beta_left[2, 0], beta_left[1, 0], beta_left[0, 0] - beta_right[0, 0]]))

                # estimating the delay based on the roots
                if np.isreal(roots).all():
                    delay = get_delay(roots)

        total_delay[k] = delay

    print(np.median(total_delay))

    # plotting the results
    plt.figure()
    plt.plot(range(total_delay.shape[0]), total_delay)
    plt.xlabel('k')
    plt.ylabel('Shift [number of samples]')
    plt.show()
