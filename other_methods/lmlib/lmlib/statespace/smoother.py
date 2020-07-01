# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
"""
Signal-Smoothing with Massage Passing

Kalman smoother - Based on the modified Bryson–Frazier message passing schema
=============================================================================

This is an implementation of a Kalman smoother according to
“On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing”
[Loeliger2016]_ ,
based on LSSMs. and the modified Bryson–Frazier message passing schema.

.. autosummary::
   :toctree: _smoother/_functions

   mbf_smoother_d


"""
from __future__ import division, absolute_import, print_function

__all__ = ["mbf_smoother_d"]

import numpy as np
from scipy.linalg import expm
import lmlib as lm


def mbf_smoother_d(lssm, sigma2U, sigma2Z, y, w=None, dt=1, n_int=1000, init_val=None):
    r"""
    This is an implementation of a Kalman smoother according to
    “On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing”
    [Loeliger2016]_ ,
    based on LSSMs and the modified Bryson–Frazier message passing schema.

    This smoother is based the LSSM


    .. math::

      X_k &= AX_{k-1} + BU_k \\
      Y_k &= CX_k + Z_k \ ,

    where :math:`A \in \mathbb{R}^{N \times N}`, :math:`B \in \mathbb{R}^{N \times M}`,
    :math:`C \in \mathbb{R}^{L \times N}`, and `k` is the time index.
    :math:`U_k \in \mathbb{R}^{M}` and :math:`Z_k \in \mathbb{R}^{L}` are independet, zero-mean Gaussian white noise
    processes with corresponding variances
    :math:`\sigma^2_U` and :math:`\sigma^2_Z`.
    This smoother is represented by the factor graph below.
    For more details see [Loeliger2016]_ .

    .. image:: /_static/mbf.PNG
       :width: 300
       :align: center
       :alt: Factor Graph of Modified Bryson–Frazier smoother

    Parameters
    ----------
    lssm : :class:`~lmlib.statespace.model.Lssm`
        Time-continuous linear state space model.
    sigma2U : float or array_like,
        shape=([K]), constant variances over all `k` :math:`\sigma^2_u \in \mathbb{R}`, or variance per `k`, :math:`\sigma^2_{u,k} \in \mathbb{R}` .
    sigma2Z : float or array_like,
        shape=([K]), constant input variances over all `k` :math:`\sigma^2_z \in \mathbb{R}`, or variance per `k`, :math:`\sigma^2_{z,k} \in \mathbb{R}` .
    y : array_like
        shape=(`K`),
        observations
    w : array_like or None
        shape=(`K`), sample weight per observation sample or `Nan` if not set.
    dt : float
        Time interval for discretezation of LSSM, Default: `dt = 1`
    n_int : int
        Accumulation elements for discretezation of LSSM, Default: `n_int = 1000`
    init_val : tuple, None
        initial state of filter (e.g., for streaming applications). The parameters are ordered like ``init_val = (m_0, V_0, xi_K, W_K)``.
        If no initial values are given. The function initiate the states as ``m_0 = np.zeros(N)``,
        ``V = 1e12 * np.eye(N)``, ``xi_K = np.zeros(N)`` and ``W_K = np.zeros(N,N)``.

    Returns
    -------
    mx : :class:`~numpy.ndarray`
        shape = `(K, N)`,
        mean state vectors.
    Vx : :class:`~numpy.ndarray`
        shape = `(K, N, N)`,
        Covariance matrix.



    --

    | |def_L|
    | |def_M_input|
    | |def_N|


    Note
    ----
    |note_MC|

    """

    # get dimensions
    y = np.asarray(y)
    K = y.shape[0]
    M = () if y.ndim is 1 else y.shape[1]
    output_shape = () if not lssm.hasVectorOutput else (lssm.C.shape[0],)

    if isinstance(sigma2U, (list, tuple, np.ndarray)):
        assert (K, M) == np.shape(sigma2U), "sigma2U and y doesn't have same shape."

    # initialization
    if lssm.hasVectorOutput:
        assert np.shape(sigma2U) == (K, lssm.input_count, lssm.input_count), (
            "The shape of sigma2U doesn't match with (K, M, M) \n"
            + "Found: sigma2U shape: "
            + str(np.shape(sigma2U))
        )
        U = sigma2U
    else:
        U = np.ones(K) * sigma2U

    if lssm.hasVectorOutput:
        assert np.shape(sigma2Z) == (K, lssm.output_count, lssm.output_count), (
            "The shape of sigma2Z doesn't match with (K, L, L) \n"
            + "Found: sigma2Z shape: "
            + str(np.shape(sigma2Z))
        )
        Z = sigma2Z
    else:
        Z = np.ones(K) * sigma2Z
        w_Z_inv = 1 / Z if w is None else w / Z

    G = np.zeros((K,) + output_shape)

    # set initial values
    if init_val is None:
        m_fw_X_init = np.zeros((lssm.N,) + output_shape)
        V_fw_X_init = np.eye(lssm.N) + 1e12
        xi_t_X_init = np.zeros((lssm.N,) + output_shape)
        W_t_X_init = np.zeros((lssm.N, lssm.N))
    else:
        m_fw_X_init, V_fw_X_init, xi_t_X_init, W_t_X_init = init_val

    V_fw_X = np.zeros((K, lssm.N, lssm.N))
    m_fw_X = np.zeros((K, lssm.N) + output_shape)
    F = np.zeros((K, lssm.N, lssm.N))
    W_t_X = np.zeros((K, lssm.N, lssm.N))
    xi_t_X = np.zeros((K, lssm.N) + output_shape)
    m_X = np.zeros((K, lssm.N) + output_shape)
    V_X = np.zeros((K, lssm.N, lssm.N))

    # identity matrix size n
    I_n = np.eye(lssm.N)

    Ad, S = _discretize(lssm, dt, n_int)

    if not lssm.hasVectorOutput:
        CTC = np.outer(lssm.C, lssm.C)

        # forward path
        m_fw_Xp_k = Ad @ m_fw_X_init
        V_fw_Xp_k = Ad @ V_fw_X_init @ Ad.T + S * U[0]
        G[0] = w_Z_inv[0] / (1 + w_Z_inv[0] * lssm.C @ V_fw_Xp_k @ lssm.C.T)
        F[0] = I_n - V_fw_Xp_k @ (CTC * G[0])

        m_fw_X[0] = m_fw_Xp_k + V_fw_Xp_k @ lssm.C.T * G[0] * (
            y[0] - lssm.C @ m_fw_Xp_k
        )
        V_fw_X[0] = (I_n - V_fw_Xp_k @ (CTC * G[0])) @ V_fw_Xp_k

        for k in range(1, K):
            m_fw_Xp_k = Ad @ m_fw_X[k - 1]
            V_fw_Xp_k = Ad @ V_fw_X[k - 1] @ Ad.T + S * U[k]

            G[k] = w_Z_inv[k] / (1 + w_Z_inv[k] * lssm.C @ V_fw_Xp_k @ lssm.C.T)
            F[k] = I_n - V_fw_Xp_k @ (CTC * G[k])

            m_fw_X[k] = m_fw_Xp_k + V_fw_Xp_k @ lssm.C.T * G[k] * (
                y[k] - lssm.C @ m_fw_Xp_k
            )
            V_fw_X[k] = (I_n - V_fw_Xp_k @ (CTC * G[k])) @ V_fw_Xp_k

        # backward path and marginal computation
        xi_t_X[K - 2] = Ad.T @ (
            F[K - 1].T @ xi_t_X_init
            + lssm.C.T * G[K - 1] * (lssm.C @ m_fw_X[K - 1] - y[K - 1])
        )
        W_t_X[K - 2] = Ad.T @ (F[K - 1].T @ W_t_X_init @ F[K - 1] + CTC * G[K - 1]) @ Ad

        m_X[K - 1] = m_fw_X[K - 1] - V_fw_X[K - 1] @ xi_t_X[K - 1]
        V_X[K - 1] = V_fw_X[K - 1] @ (I_n - W_t_X[K - 1] @ V_fw_X[K - 1])

        for k in reversed(range(K - 1)):
            xi_t_X[k - 1] = Ad.T @ (
                F[k].T @ xi_t_X[k] + lssm.C.T * G[k] * (lssm.C @ m_fw_X[k] - y[k])
            )
            W_t_X[k - 1] = Ad.T @ (F[k].T @ W_t_X[k] @ F[k] + CTC * G[k]) @ Ad
            # marginals
            m_X[k] = m_fw_X[k] - V_fw_X[k] @ xi_t_X[k]
            V_X[k] = V_fw_X[k] @ (I_n - W_t_X[k] @ V_fw_X[k])
            # m_X[k] = m_fw_X[k] - V_fw_X[k] @ xi_t_X[k]
            # V_X[k] = V_fw_X[k] @ W_t_X[k] @ V_fw_X[k]
    else:

        assert (
            False
        ), "Not multi-output LSSM not yet tested. Use single-output LSSM instead."

        # forward path
        m_fw_Xp_k = Ad @ m_fw_X_init
        V_fw_Xp_k = Ad @ V_fw_X_init @ Ad.T + S * U[0]
        G[0] = 1 / (Z[0] + lssm.C @ V_fw_Xp_k @ lssm.C.T)
        F[0] = I_n - V_fw_Xp_k @ lssm.C.T @ G[0] @ lssm.C

        m_fw_X[0] = m_fw_Xp_k + V_fw_Xp_k @ lssm.C.T * G[0] * (
            y[0] - lssm.C @ m_fw_Xp_k
        )
        V_fw_X[0] = (I_n - V_fw_Xp_k @ lssm.C.T @ G[0] @ lssm.C) @ V_fw_Xp_k

        for k in range(1, K):
            m_fw_Xp_k = Ad @ m_fw_X[k - 1]
            V_fw_Xp_k = Ad @ V_fw_X[k - 1] @ Ad.T + S * U[k]

            G[k] = np.linalg.pinv(Z[k] + lssm.C @ V_fw_Xp_k @ lssm.C.T)
            F[k] = I_n - V_fw_Xp_k @ lssm.C.T @ G[k] @ lssm.C

            m_fw_X[k] = m_fw_Xp_k + V_fw_Xp_k @ lssm.C.T * G[k] * (
                y[k] - lssm.C @ m_fw_Xp_k
            )
            V_fw_X[k] = (I_n - V_fw_Xp_k @ lssm.C.T @ G[k] @ lssm.C) @ V_fw_Xp_k

        # backward path and marginal computation
        xi_t_X[K - 2] = Ad.T @ (
            F[K - 1].T @ xi_t_X_init
            + lssm.C.T * G[K - 1] * (lssm.C @ m_fw_X[K - 1] - y[K - 1])
        )
        W_t_X[K - 2] = (
            Ad.T
            @ (F[K - 1].T @ W_t_X_init @ F[K - 1] + lssm.C.T @ G[K - 1] @ lssm.C)
            @ Ad
        )

        m_X[K - 1] = m_fw_X[K - 1] - V_fw_X[K - 1] @ xi_t_X[K - 1]
        V_X[K - 1] = V_fw_X[K - 1] @ (I_n - W_t_X[K - 1] @ V_fw_X[K - 1])

        for k in reversed(range(K - 1)):
            xi_t_X[k - 1] = Ad.T @ (
                F[k].T @ xi_t_X[k] + lssm.C.T * G[k] * (lssm.C @ m_fw_X[k] - y[k])
            )
            W_t_X[k - 1] = (
                Ad.T @ (F[k].T @ W_t_X[k] @ F[k] + lssm.C.T @ G[k] @ lssm.C) @ Ad
            )

            # marginals
            m_X[k] = m_fw_X[k] - V_fw_X[k] @ xi_t_X[k]
            V_X[k] = V_fw_X[k] @ (I_n - W_t_X[k] @ V_fw_X[k])

    return m_X, V_X


def _discretize(lssm, dt, N):
    """
    Returns a discrete LSSM or ALSSM

    Parameters
    ----------
    lssm : :class:`~lmlib.statespace.lssm.LSSM`
        Linear state space model
    dt : float
        Time interval
    N : int
        integration pieces

    Returns
    -------
    d_lssm : :class:`~lmlib.statespace.lssm.LSSM`
        discretize LSSM
    """
    A = expm(lssm.A * dt)
    eAtN = expm(A * (dt / N) * 0.5)
    eAt1 = expm(A * (dt / N))
    sep = np.zeros_like(A)
    B = np.outer(lssm.B, lssm.B)
    for _ in range(N):
        sep += eAtN @ B @ eAtN.T
        eAtN = eAtN @ eAt1
    sep = sep / N
    return A, sep
