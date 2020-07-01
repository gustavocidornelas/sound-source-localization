.. _lmlib_about:

About lmlib
===========
lmlib (localized model library) is a signal processing library providing methods to perform singal processing tasks such as event detection, signal separation, pattern identification in single- or multi-channel signals.
The used methods are based on localized models using autonomous linear state-space and polynomial models.
Localization is done by time-domain windowing.

Most of the proposed methods take use of efficient, recursive computations.
However, all methods in this library are of rather low computationally complexity, which typically scales linearly with the number of samples.
Therefore, the methods are also suitable for low-power platforms or fast processing.
Most of the methods are non-iterative, i.e., provide a fixed execution run-time.


Methods provided by lmlib are mostly derived or modified at the Signal and Information Processing Laboratory (ISI),
ETH Zurich.
Full references are given in the API documentation in the referene list.


lmlib is written in python and is published open-source under XXX-licence.

