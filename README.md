# Sound source localization via onset detection

The code for this project is organized in 3 separate directories, namely, `onset-detection`, `local-polynomial-fitting` and `delay-estimation`. The code contained in each directory corresponds to the steps followed by the proposed algorithm to go from the raw multi-channel audio signal to the delay estimates.

## Onset detection
In the `onset-detection` directory, the main script to be executed is `main.py`. This script is responsible for performing the onset model fit to the raw audio signals and saving the computed Local cost ratios (LCRs) in the end. Most of the parameters of the fit, such as the onset model, the window model, the sampling frequency, etc. are defined in the `parameter.py` file. The files  `onset_models.py` and `window_models.py` contain the LSSM representations of the onset models and window models considered, respectively, and are only used by the main script. The parameters which we tune, namely the onset decay, window decay and onset model frequency are passed as arguments when running `main.py`
 
