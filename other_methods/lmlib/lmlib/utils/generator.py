# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
r"""

Provides deterministic, random signal generators and static signals.

Deterministic Signal Generators
-------------------------------
- Sinusoidal : :meth:`gen_sinusoidal`
- Rectangle : :meth:`gen_rectangle`
- Triangle : :meth:`gen_triangle`
- Ramp : :meth:`gen_ramp`
- Unit impulse : :meth:`gen_unit_impulse`
- Sinusoidal baseline : :meth:`gen_baseline_sin`
- Exponential decay : :math:`gen_exponential`

Random Signal Generators
------------------------
- White Gaussian noise : :meth:`gen_wgn`
- Random ramp : :meth:`gen_rand_ramps`
- Random pulse : :meth:`gen_rand_pulse`
- Random unit impulse : :meth:`gen_rand_unit_impulse`
- Random walk : :meth:`gen_rand_walk`

Static Signals
--------------
- Loading single-channel signals : :meth:`load_single_channel`
- Loading multi-channel signals : :meth:`load_multi_channel`

For the static signals names, see :ref:`lmlib_signal_catalog`.

Signal Modifier Functions
-------------------------
- Convolve signals :meth:`gen_convolve`
- Create Multi-Channel signals :meth:`gen_multichannel`

The difference on single- and multi-channel signals is described in :ref:`lmlib_single_multi_channel`.
"""

from __future__ import division, absolute_import, print_function

__all__ = [
    "gen_sinusoidal",
    "gen_rectangle",
    "gen_triangle",
    "gen_ramp",
    "gen_unit_impulse",
    "gen_baseline_sin",
    "gen_exponential",
    "gen_wgn",
    "gen_rand_ramps",
    "gen_rand_pulse",
    "gen_rand_unit_impulse",
    "gen_rand_walk",
    "load_single_channel",
    "load_multi_channel",
    "gen_convolve",
    "gen_multichannel",
]

import numpy as np
import lmlib as lm

data_path = lm.abs_path + "/utils/data/"


def gen_sinusoidal(K, k_period):
    """Sinusoidal signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        Normalized period in samples per cycle.

    Returns
    -------
    out : class:`numpy.ndarray`
        Returns a sinusoidal signal with length `K` with `k_period` samples per cycle.
    """
    return np.sin(2 * np.pi * np.linspace(0, K / k_period, K))


def gen_exponential(K, decay, k=0):
    """

    Parameters
    ----------
    K : int
        Signal length
    decay : float
        Decay factor
    k : int
        Initial value 1 location

    Returns
    -------
    out : class:`numpy.ndarray`
        Returns an exponential function  of length K and decay with initial value 1 at location k
    """
    return np.power(decay, np.arange(0 - k, K - k))


def gen_rectangle(K, k_period, k_on):
    """Rectangular signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        Normalized period in samples per cycle.
    k_on : int
        First `p_on` samples of the period are one else zero.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a rectangular signal with length `K` with `k_period` samples per period where the first `p_on` sample are one.
    """
    assert k_on <= k_period, "k_on must be smaller equal than period."
    out = np.zeros(K,)
    for p in np.arange(0, K, k_period):
        out[p + range(k_on)] = np.ones((k_on,))
    return out


def gen_ramp(K, k_period):
    """Ramp signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        Normalized period in samples per cycle.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a repetitive ramp signal with length `K` with `k_period` samples per ramp from 0 to 1.
    """
    return np.remainder(range(K), k_period) / k_period


def gen_triangle(K, k_period):
    """Triangular signal generator

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        Normalized period in samples per cycle.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a triangular signal with length `K` with `k_period` samples per triangle.
    """
    return 1 - abs(np.remainder(range(K), k_period) - 0.5 * k_period) / (0.5 * k_period)


def gen_unit_impulse(K, k):
    """Unit impulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    k : int, list
        Index for unit impulse

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a unit impulse signal with length `K` where the signal at the index/indices in `k` is one.
    """
    out = np.zeros((K,))
    if isinstance(k, int):
        assert 0 <= k < K, "k is not in the range of K."
    else:
        assert all(0 <= k0 < K for k0 in k), "k is not in the range of K."
    out[k] = 1
    return out


def gen_baseline_sin(K, k_period):
    """Baseline signal generator shaped by 4 sinusoidal.

    The baseline is formed by the product of 4 sinusoidal.
    The highest frequency is given py `k_period` samples per cycle.
    The lower 3 frequencies are linear spaced from k_periodto K.
    Each sinusoidal is weighted with its samples/period divided by `K`.

    Parameters
    ----------
    K : int
        Signal length
    k_period: int
        Normalized period of the highest frequency in the baseline in samples per cycle.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a baseline signal shaped by 4 sinusoidal of length `K`.
    """
    out = np.ones((K,))
    for p in np.linspace(K, k_period, 4):
        out *= p / K * gen_sinusoidal(K, p)
    return out


def gen_wgn(K, sigma, seed=None):
    """Triangular signal generator

    Parameters
    ----------
    K : int
        Signal length
    sigma : float
        Variance
    seed : int, None
        Seed number for random number generator.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns white Gaussian noise with length `K`  and variance `sigma`.
    """
    np.random.seed(seed)
    return np.random.normal(0, sigma, K)


def gen_rand_ramps(K, N, seed=None):
    """Random ramps signal generator

    The first edge/kink in the signal is by ``np.floor(0.05 * K)`` and the last at ``np.ceil(0.95 * K)``.
    The first and last segment aren't included in the number of ramps `N`.

    .. note::
       If seed is None :meth:`gen_rand_ramps()` searches for a signal where the edges are at least
       ``np.floor(0.05 * K)`` samples apart.


    Parameters
    ----------
    K : int
        Signal length
    N : int
        Number of Ramps in the signal
    seed : int, None
        Seed number for random number generator.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns `N` random ramps in a signal of length `K`.
    """
    np.random.seed(seed)
    k = np.sort(np.random.uniform(low=0, high=K, size=N + 1).astype(int))
    y = np.random.wald(0.5, 1, (N + 1))
    sign = np.sign(np.random.randn((N + 1)))
    return np.interp(np.arange(K), k, sign * y)


def gen_rand_pulse(K, N, length, seed=None):
    """Random pulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    N : int
        Number of pulses in the signal
    length : int
        Length of Samples (on-samples)
    seed : int, None
        Seed number for random number generator.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns `N` unity pulses at random positions in a signal of length `K`.
    """
    np.random.seed(seed)
    rui = gen_rand_unit_impulse(K, N, seed=seed)
    return np.convolve(rui, np.ones((length,)), "same")


def gen_rand_unit_impulse(K, N, seed=None):
    """Random pulse signal generator

    Parameters
    ----------
    K : int
        Signal length
    N : int
        Number of impulses in the signal
    seed : int, None
        Seed number for random number generator.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns `N` unity impulses at random positions in a signal of length `K`.
    """
    np.random.seed(seed)
    k = np.random.randint(np.floor(0.05 * K), np.ceil(0.95 * K), N)
    return gen_unit_impulse(K, k)


def gen_rand_walk(K, seed=None):
    """Random walk generator

    Parameters
    ----------
    K : int
        Signal length
    seed : int, None
        Seed number for random number generator.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Returns a random walk of length `K`.
    """
    np.random.seed(seed)
    return np.cumsum(np.sign(np.random.randn(K)) * np.random.ranf(K,))


def load_single_channel(name, K, k=0, ch_select=0):
    """Loading a single channel signal

    Parameters
    ----------
    name : str
        Signal name. See :ref:`lmlib_signal_catalog`
    K : int
        Signal length as positive integer value or -1 for load to the end of the file.
    k : int
        Signal start index.
        If the signal length is smaller than `K` samples, due to a to large start index `k`, an assertion is raised.
    ch_select : int
        Select channel number if file has multiple channels. Note that `` 0 <= channel < M``,
        where ``M`` is the number of channel in the filename.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Single-channel signal with shape=(K,)

    Examples
    --------
    >>> from lmlib.utils.generator import load_multi_channel
    >>> y = load_multi_channel('EECG_BASELINE_1CH_10S_FS2400HZ.csv' K=6000)
    >>> print(y.shape)
    (6000,)
    """
    assert isinstance(name, str), "Filename is not a string."

    filepath = data_path + name
    y = np.loadtxt(filepath, delimiter=",")

    if y.ndim is 2:
        K_file, M = y.shape
        assert 0 <= ch_select < M, (
            "The channel selection is out of range."
            + "\nExpected range: [0, {})".format(M)
            + "\nFound ch_select: {}".format(ch_select)
        )
        if k is 0:
            if K is not -1:
                assert 0 <= K < K_file, (
                    "The signal length K out of range."
                    + "\nExpected range: [0, {})".format(K_file)
                    + "\nFound K: {}".format(K)
                )
        else:
            assert 0 <= k < K_file, (
                "The signal start index k out of range."
                + "\nExpected range: [0, {})".format(K_file)
                + "\nFound k: {}".format(k)
            )
            if K is not -1:
                assert 0 <= k + K < K_file, (
                    "The signal start with signel length out of range."
                    + "\nExpected range: [0, {})".format(K_file)
                    + "\nReduce k to : {}".format(K_file - K - 1)
                    + "\nor reduce K to : {}".format(K_file - k - 1)
                    + "\nFound k+K: {}".format(k + K)
                )
        return (
            np.squeeze(y[k::, ch_select])
            if K is -1
            else np.squeeze(y[k : k + K, ch_select])
        )
    elif y.ndim is 1:
        K_file = y.size
        if k is 0:
            if K is not -1:
                assert 0 <= K < K_file, (
                    "The signal length K out of range."
                    + "\nExpected range: [0, {})".format(K_file)
                    + "\nFound K: {}".format(K)
                )
        else:
            assert 0 <= k < K_file, (
                "The signal start index k out of range."
                + "\nExpected range: [0, {})".format(K_file)
                + "\nFound k: {}".format(k)
            )
            if K is not -1:
                assert 0 <= k + K < K_file, (
                    "The signal start with signel length out of range."
                    + "\nExpected range: [0, {})".format(K_file)
                    + "\nReduce k to : {}".format(K_file - K - 1)
                    + "\nor reduce K to : {}".format(K_file - k - 1)
                    + "\nFound k+K: {}".format(k + K)
                )

        return np.squeeze(y[k::]) if K is -1 else np.squeeze(y[k : k + K])


def load_multi_channel(name, K, k=0, ch_select=None):
    """Loading a multi channel signal

    Parameters
    ----------
    name : str
        Signal name. See :ref:`lmlib_signal_catalog`
    K : int
        Signal length as positive integer value or -1 for load to the end of the file.
    k : int
        Signal start index.
        If the signal length is smaller than `K` samples, due to a to large start index `k`, an assertion is raised.
    ch_select : list, None
        Select channel number if file has multiple channels. Note that `` 0 <= channel < M``,
        where ``M`` is the number of channel in the filename.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        Multi-channel signal with shape=(K,M), where M is the number of channels.

    Examples
    --------
    >>> from lmlib.utils.generator import load_multi_channel
    >>> y = load_multi_channel('EECG_BASELINE_9CH_10S_FS2400HZ.csv' K=5000)
    >>> print(y.shape)
    (5000, 9)
    """
    if ch_select is None:
        ch_select = []
    assert isinstance(name, str), "Filename is not a string."

    filepath = data_path + name
    y = np.loadtxt(filepath, delimiter=",")
    assert y.ndim is 2, "File contains no multichannel signal.\nCheck filename: " + name
    K_file, M = y.shape
    for ch in ch_select:
        assert 0 <= ch < M, (
            "The channel selection is out of range."
            + "\nExpected range: [0, {})".format(M)
            + "\nFound channel in ch_select: {}".format(ch)
        )
    if k is 0:
        if K is not -1:
            assert 0 <= K < K_file, (
                "The signal length K out of range."
                + "\nExpected range: [0, {})".format(K_file)
                + "\nFound K: {}".format(K)
            )
    else:
        assert 0 <= k < K_file, (
            "The signal start index k out of range."
            + "\nExpected range: [0, {})".format(K_file)
            + "\nFound k: {}".format(k)
        )
        if K is not -1:
            assert 0 <= k + K < K_file, (
                "The signal start with signel length out of range."
                + "\nExpected range: [0, {})".format(K_file)
                + "\nReduce k to : {}".format(K_file - K - 1)
                + "\nor reduce K to : {}".format(K_file - k - 1)
                + "\nFound k+K: {}".format(k + K)
            )
    if not ch_select:
        return y[k::] if K is -1 else y[k : k + K]
    return (
        np.squeeze(y[k::, ch_select])
        if K is -1
        else np.squeeze(y[k : k + K, ch_select])
    )


def gen_convolve(y1, y2):
    """
    Convolve two signals. The output shape remains the shape of `y1`.

    Parameters
    ----------
    y1 : array_like
        This signal is the base signal. It can be single- ord multi-channel.
    y2 : array_like
        Signal template contains the a signal which gets convolved with `y1`.

    Returns
    -------
    out : :class:`~numpy.ndarray`
        The output is the convolution of the two signal where the output has the same shape as `y1`. The convolution
        is applied to all channels.
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    assert y1.ndim <= 2, "y1 has more then two-dimensions."
    assert y2.ndim < 2, "y2 has more then one-dimension."

    if y1.ndim is 2:
        return np.stack(*[np.convolve(ych, y2, "same") for ych in y1], axis=-1)
    return np.convolve(y1, y2, "same")


def gen_multichannel(arrays, option="multi-output"):
    """
    Returns a multi-channel signal out of arrays.
    Either the signal is multi-output of shape ``(K, L)`` where L is the number of channels.
    Or the signal has multi-observations (parallel) which has a shape of ``(K, 1, O)``,
    where `O` is the number if channels.

    Parameters
    ----------
    arrays : tuple
        Set of single-channel arrays
    option : str
        Use ``'multi-output'`` for a signal shape of ``(K, L)`` or
        ``'multi-observation'`` for a signal shape of ``(K, 1, O)``.

    Returns
    -------
    signal : :class:`~numpy.ndarray`
        Multi-channel signal
    """
    y = np.column_stack(arrays)
    if option == "multi-output":
        return y
    elif option == "multi-observation":
        return y[:, np.newaxis, :]
    else:
        raise ValueError(
            "Unknown option. \n Expect: 'multi-output' or 'multi-observation'\n Found: {}".format(
                option
            )
        )
