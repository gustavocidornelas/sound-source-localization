"""Helper functions and miscellaneous
"""

from __future__ import division, absolute_import, print_function

__all__ = ["matrix_power", "kron_sequence"]

import numpy as np
import lmlib as lm


def matrix_power(A, i):
    """Calculate the `i`th power of matrix `A`.

    Parameters
    ----------
    A : array_like
        Two-dimensional matrix
    i : float or array_like
        matrix exponent or list of matrix exponents

    Returns
    -------
    out : :class:`~numpy.ndarray`
          shape ``(len(i), shape(A))``
          matrix power with
    """
    if np.isscalar(i) or np.ndim(i) == 0:
        return np.linalg.matrix_power(A, i)
    return np.stack([np.linalg.matrix_power(A, i_) for i_ in i], axis=0)


def kron_sequence(arrays):
    """
    Kroneker product of a sequence of matrices

    .. math::
        B = A_0 \otimes A_1 \otimes A_2 \otimes A_3 \otimes \dots A_N

    Parameters
    ----------
    arrays :  tuple of array_like
        sequence of matrices

    Returns
    -------
    prod : array_like, int
        Kronecker product of sequence of matrices

    Examples
    --------
    >>> a1 = np.array([[2, 3, 4], [0, 1, 4]])
    >>> a2 = np.array([[9, 0], [1, 1]])
    >>> a3 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> b = kron_sequence((a1, a2, a3))
    >>> print(b)
    [[  0  18  36   0   0   0   0  27  54   0   0   0   0  36  72   0   0   0]
     [ 54  72  90   0   0   0  81 108 135   0   0   0 108 144 180   0   0   0]
     [108 126 144   0   0   0 162 189 216   0   0   0 216 252 288   0   0   0]
     [  0   2   4   0   2   4   0   3   6   0   3   6   0   4   8   0   4   8]
     [  6   8  10   6   8  10   9  12  15   9  12  15  12  16  20  12  16  20]
     [ 12  14  16  12  14  16  18  21  24  18  21  24  24  28  32  24  28  32]
     [  0   0   0   0   0   0   0   9  18   0   0   0   0  36  72   0   0   0]
     [  0   0   0   0   0   0  27  36  45   0   0   0 108 144 180   0   0   0]
     [  0   0   0   0   0   0  54  63  72   0   0   0 216 252 288   0   0   0]
     [  0   0   0   0   0   0   0   1   2   0   1   2   0   4   8   0   4   8]
     [  0   0   0   0   0   0   3   4   5   3   4   5  12  16  20  12  16  20]
     [  0   0   0   0   0   0   6   7   8   6   7   8  24  28  32  24  28  32]]
    """

    if len(arrays) == 0:
        return 1
    if len(arrays) == 1:
        return arrays[0]
    if len(arrays) > 1:
        return np.kron(arrays[0], kron_sequence(arrays[1::]))
