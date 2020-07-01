.. _lmlib_single_multi_channel:

Single- and Multi-Channel Signals
=================================

- Single-channel signals is termed for signal array which are **one-dimensional**.
  For example:

  .. code::

     y = [0.10, 0.41, 0.61, 1.22, 4.01, ..., -3,44, 1,76]

  The single-channel signal ``y`` is a one-dimensional array with a shape of ``(K,)``, where `K` determines  number of samples.

- Multi-channel signals is termed for signal array which are **multi-dimensional** (normally two-dimensional).
  For example:

  .. code::

     y = [[0.65, 0.31], [1.51, 1.22], [-2.97, 0.11], ..., [-3.44, 1.76], [4.66, 8.12]]

  The multi-channel signal ``y`` is a two-dimensional array with a shape of ``(K, M)``, where `K` determines  number of samples and `M` the number of channels.

.. note::

   Its also possible to have a multi-channel signal with **one** signal.
   The multi-channel signal ``y`` is still a two-dimensional array with a shape of ``(K, M)``, where `M` is 1.

