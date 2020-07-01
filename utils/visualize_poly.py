import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # reading the LCR samples
    LCR = np.genfromtxt(
        "/Users/gustavocidornelas/Desktop/sound-source/decaying_sinusoid_0.99_gamma_0.999_Az_90_freq_"
        "80.0.csv",
        delimiter=",",
    )
    LCR_L = LCR[501:, 1]
    LCR_R = LCR[501:, 2]

    # separating the samples of interest
    LCR_L = LCR_L[242000 : 242000 + 50]
    LCR_R = LCR_R[242000 : 242000 + 50]

    # reading the coefficients (already transformed to the canonical basis)
    coeff_L = np.genfromtxt(
        "transformed_poly_fit_result_LEFT.csv", skip_header=False, delimiter=","
    )
    coeff_L = coeff_L[242000, :]
    coeff_R = np.genfromtxt(
        "transformed_poly_fit_result_RIGHT.csv", skip_header=False, delimiter=","
    )
    coeff_R = coeff_R[242000, :]

    l = range(0, 50, 1)

    # local polynomial fits
    y_L = coeff_L[0] + coeff_L[1] * l + coeff_L[2] * l * l + coeff_L[3] * l * l * l
    y_R = coeff_R[0] + coeff_R[1] * l + coeff_R[2] * l * l + coeff_R[3] * l * l * l

    plt.figure()
    plt.plot(l, y_L)
    plt.plot(l, y_R)
    plt.xlabel("k")
    plt.title("Local polynomial fit")
    plt.figure()
    plt.plot(range(LCR_L.shape[0]), LCR_L)
    plt.plot(range(LCR_R.shape[0]), LCR_R)
    plt.xlabel("k")
    plt.title("LCR")
    plt.show()
