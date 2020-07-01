import os
import numpy as np

if __name__ == "__main__":

    # frequency
    f_min = 3000.0
    f_max = 5000.0
    f_inc = 100.0

    # onset decay
    onset_decay_min = 0.9
    onset_decay_max = 1.0
    onset_decay_inc = 0.01

    # window decay
    window_decay_min = 0.9
    window_decay_max = 1.0
    window_decay_inc = 0.01

    for freq in np.arange(f_min, f_max, f_inc):
        for onset_dec in np.arange(onset_decay_min, onset_decay_max, onset_decay_inc):
            for window_dec in np.arange(
                window_decay_min, window_decay_max, window_decay_inc
            ):
                print(
                    "Running new point: "
                    + str(freq)
                    + " "
                    + str(onset_dec)
                    + " "
                    + str(window_dec)
                )
                os.system(
                    "python main.py "
                    + str(freq)
                    + " "
                    + str(onset_dec)
                    + " "
                    + str(window_dec)
                )
