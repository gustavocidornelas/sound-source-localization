
# Sound source localization via onset detection

## About
Humans can perform sound source localization in a seemingly effortless manner. To do so, one of the main cues we rely on is the interaural time difference (ITD), which is related to the perceived time delay between the versions of the audio signal received by the right and left ears. There are many computational methods that strive to estimate the ITD from binaural audio signals and this repository contains the code for an algorithm we propose based on fitting autonomous linear state space models (LSSM).

This repository was developed as part of my master's thesis (Spring 2020) and semester project (Fall 2019) at the [Signal and Information Processing Laboratory (ISI)](https://isi.ee.ethz.ch/), at [ETH Zürich](https://www.ethz.ch/en.html). The full reports with the detailed description of our approach is avaliable [here](#references).

- [Requirements](#requirements)
- [Project structure](#project-structure)
- [References](#references)

## Requirements
- Python 3 (tested with Python 3.7)
- NumPy (tested with version 1.19.1)

## Project structure

### `/model` directory
The code for this project is organized in modules, inside the `/model` directory. 

The `lcr_fit` module contains the methods used to fit the onset models to the binaural audio signals, producing the local cost ratios (LCRs). This model is pretty flexible when it comes to the onset models of choice. One can choose between fitting decaying sinusoid or gammatone filters onset models using exponential or gamma windows. 

The `polynomial_fit` module has the methods to locally fit 3rd degree polynomials to the LCR pairs using rectangular windows.

The `delay_estimation` module contains the classes and methods responsible for estimating the time delay between the 3rd degree polynomial fits. Some of the classes in this module implement the delay estimation procedure for the single frequency case while others implement the versions that work with the filter bank approach.

Some examples of how to import each module and how to make them work can be found in the scripts `run_filter_bank_dataset.py`, `run_filter_bank_sig_stroke_dataset.py` and `run_linear_combination_dataset.py`.

### `/online_demo` directory
The `/online_demo` directory contains an online demo of the algorithm for the single frequency case. The code here was developed for simple illustration purposes, to actually observe the algorithm running in an online manner.

### `/other_methods` directory
The `/other_methods` directory contains some of the implementation of the classical ITD estimation methods that we use as baselines for comparison of our model. Some of the methods are the threshold methods, cross-correlation methods and group delay methods (implemented on MATLAB). 

### `/utils` directory
This directory is a complete mess. Sorry about that. It contains some of the code that I used to visualize some preliminary results and to debug my implementation. The code here is poorly documented and very inefficient, so I discourage you from using it.  

## References
G. C. Ornelas, ["Robust time delay estimation via onset detection filter bank"](msc_thesis.pdf), Master's thesis, ETH Zurich, Switzerland, 2020.

G. C. Ornelas, ["Sound source localization via onset detection"](semester_project.pdf), Semester thesis, ETH Zurich, Switzerland, 2019.
 
