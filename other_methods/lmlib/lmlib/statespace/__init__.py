r"""
This module provides methods to define linear state space models and to use them as signal models in least squares
problems.
The computational complexity of this module is, for many standard task, rather low as it takes advantage of recursive
computation rules.

This module implements the methods published in  [Zalmai2017]_, [Wildhaber2018]_, and [Wildhaber2019]_.

This module contains:
---------------------

* classes to set up basic linear state space models (LSSMs) :class:`~lmlib.statespace.model.Lssm` and
  autonomous linear state space models (ALSSMs) :class:`~lmlib.statespace.model.Alssm`.

* modifier classes to merge multiple linked LSSMs into one single system, e.g. for

    * multiple systems with a common input,
    * multiple system where the output of interest is given by the sum of the single outputs,
    * multiple system where the output of interest is given by the product of the single outputs,
    * etc.

Base class for such merged systems is :class:`~lmlib.statespace.model.AlssmContainer` and
:class:`~lmlib.statespace.model.LssmContainer`.

* recursive computation rules to efficiently evaluate and minimize cost functions built on
  such linear state space models. These cost functions along with the recursive computation rules are implemented in
  :class:`~lmlib.statespace.costfunc.SEParam`.

* Kalman filters (Kalman smoother) based on linear state space models.

Submodules
----------
.. autosummary::
   :toctree: _statespace
   :template: submodule.rst

   model
   costfunc
   smoother
   utils

"""
from __future__ import division, absolute_import, print_function

from lmlib.statespace.model import *
from lmlib.statespace.costfunc import *
from lmlib.statespace.smoother import *
from lmlib.statespace.utils import *
