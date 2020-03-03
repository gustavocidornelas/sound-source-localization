# Sound source localization via onset detection

The code for this project is organized in 3 separate directories, namely, `/onset-detection`, `/local-polynomial-fitting` and `/delay-estimation`. The code contained in each directory corresponds to the steps followed by the proposed algorithm to go from the raw multi-channel audio signal to the delay estimates.

## Onset detection
In the `/onset-detection` directory, the main script to be executed is `main.py`. This script is responsible for performing the onset model fit to the raw audio signals and saving the computed Local cost ratios (LCRs) in the end as a csv file. 

Most of the parameters of the fit, such as the onset model, the window model, the sampling frequency, etc. are defined in the `parameter.py` file. The files  `onset_models.py` and `window_models.py` contain the LSSM representations of the onset models and window models considered, respectively, and are only used by the main script. The parameters which we tune, namely the onset decay, window decay and onset model frequency are passed as arguments when running `main.py`. For example, to fit the onset model with frequency equal to 80 Hz with onset decay equal to 0.99 and window decay equal to 0.987, one should use the following command in the terminal:

`python main.py 80 0.99 0.987 /audio_data_directory`
 
 where `/audio_data_directory` is the full path to the directory containing the audio files. The onset model (i.e., 'decaying_sinusoid' or 'gammatone') and window model (i.e., 'gamma' or 'exponential') are defined in the `parameters.py` file.

## Local polynomial fitting
In the `/local-polynomial-fitting` directory there are two scripts: `fit_poly_rect.py` and `fit_poly_exp.py`. The `fit_poly_rect.py` locally fits a 3rd degree polynomial to the LCRs obtained in the previous step using a rectangular window. The lenght of the rectangular window used is defined inside `fit_poly_rect.py`. After running `fit_poly_rect.py`, the local polynomial fit coefficients are saved in a csv file, already in the canonical basis. The LCR used for the fit (i.e., the right or the left LCR) is chosen via one of the parameters passed when executing the script. To run such script, one should use the following command in the terminal: 

`python fit_poly_rect.py /LCR_data_directory index`
  
 where `/LCR_data_directory` is the full path to the directory containing the LCR files and `index` assumes values of 1 (for the LCR from the left) or 2 (for the LCR from the right).
 
 The script `fit_poly_exp.py` performs the fit using an exponential window. This script was used for experimentation purposes only, and does not save the polynomial fit results in the end. The message passing algorithm used in it, though, can be useful for reference.
 
 ## Delay estimation
 The only script present in the `/delay-estimation` directory is `delay_estimation.py`. This script performs the delay estimation given the local polynomial fits for the LCRs from the left and from the right. The threshold parameters used for determining if the LCRs are rising are defined inside the script. It saves, then, a csv file containing all the unique delay estimated. The script receives the path to each polynomial coefficients file, when called as in: 
 
 `python delay_estimation.py /coeff_from_left_file /coeff_from_right_file `
 
 ## Full algorithm
 The whole algorithm can be executed end-to-end by running the script `master_script.py`. The algorithm starts by generating the LCRs from the raw audio signals, then, locally fitting the 3rd degree polynomials to the LCRs and finally estimating the delays and saving them in a csv file. The script `master_script.py` can be used as a good reference to the syntax used to call each part of the algorithm separately.
 
 
