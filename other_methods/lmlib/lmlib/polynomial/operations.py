from __future__ import division, absolute_import, print_function

__all__ = [
    "poly_sum",
    "poly_sum_coef",
    "poly_sum_coef_Ls",
    "poly_sum_expo",
    "poly_sum_expo_Ms",
    "poly_prod",
    "poly_prod_coef",
    "poly_prod_expo_Ms",
    "poly_prod_expo",
    "poly_square",
    "poly_square_coef",
    "poly_square_expo",
    "poly_square_expo_M",
    "poly_shift",
    "poly_shift_coef",
    "poly_shift_coef_L",
    "poly_shift_expo",
    "poly_dilation",
    "poly_dilation_coef",
    "poly_dilation_coef_L",
    "poly_int",
    "poly_int_coef",
    "poly_int_coef_L",
    "poly_int_expo",
    "poly_diff",
    "poly_diff_coef",
    "poly_diff_coef_L",
    "poly_diff_expo",
    "mpoly_add",
    "mpoly_add_coefs",
    "mpoly_add_expos",
    "mpoly_multiply",
    "mpoly_prod",
    "mpoly_square",
    "mpoly_square_coef",
    "mpoly_square_expos",
    "mpoly_shift",
    "mpoly_shift_coef",
    "mpoly_shift_coef_L",
    "mpoly_shift_expos",
    "mpoly_int",
    "mpoly_int_coef",
    "mpoly_int_coef_L",
    "mpoly_int_expos",
    "mpoly_def_int",
    "mpoly_def_int_coef",
    "mpoly_def_int_coef_L",
    "mpoly_def_int_expos",
    "mpoly_substitute",
    "mpoly_substitute_coef",
    "mpoly_substitute_coef_L",
    "mpoly_substitute_expos",
    "mpoly_dilate",
    "mpoly_dilate_coefs",
    "mpoly_dilate_coef_L",
    "mpoly_dilate_expos",
    "mpoly_dilate_ind",
    "mpoly_dilate_ind_coefs",
    "mpoly_dilate_ind_coef_L",
    "mpoly_dilate_ind_expos",
    "permutation_matrix_square",
    "permutation_matrix",
    "commutation_matrix",
]

import numpy as np
from scipy.special import comb
import lmlib as lm


def poly_sum(polys):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^q}`

    Sum of univariate polynomials ``Poly(alpha,q),... , Poly(beta,r)``, all of common variable x



    Parameters
    ----------
    polys : tuple of :class:`~lmlib.polynomial.poly.Poly`
        ``(Poly(alpha,q),... , Poly(beta,r))``

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :class:`Poly`, ``Poly(alpha_tilde, q_tilde)`` - sum of elements listed in `polys`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)

    """
    return lm.Poly(poly_sum_coef(polys), poly_sum_expo([poly.expo for poly in polys]))


def poly_sum_coef(polys):
    r""" :math:`\alpha^\mathsf{T} x^q + \dots + \beta^\mathsf{T} x^q = \color{blue}{\tilde{\alpha}}^\mathsf{T} x^q`

    Coefficient vector of sum of univariate polynomials in `polys`, all of common variable x

    Parameters
    ----------
    polys : tuple of :class:`~lmlib.polynomial.poly.Poly`
        ``(Poly(alpha,q),... , Poly(beta,r))``

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        ``alpha_tilde`` - Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)

    """
    Ls = poly_sum_coef_Ls([poly.expo for poly in polys])
    return np.sum(
        [np.dot(L, coef) for L, coef in zip(Ls, [poly.coef for poly in polys])], axis=0
    )


def poly_sum_coef_Ls(expos):
    r""" :math:`(\color{blue}{\Lambda_1} \alpha + \dots + \color{blue}{\Lambda_2}\beta)^\mathsf{T} x^{\tilde{q}}`

    Exponent Manipulation Matrices of :func:`poly_sum`

    Parameters
    ----------
    expos : tuple of array_like
        Exponent vectors :math:`q, r`

    Returns
    -------
    Ls : list of :class:`~numpy.ndarray`
        Coefficient Manipulation Matrices :math:`\Lambda_1, \Lambda_2,\dots`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    indices_or_sections = np.cumsum([len(expo) for expo in expos[0:-1]])
    Q = sum(len(expo) for expo in expos)
    return np.split(np.eye(Q), indices_or_sections, axis=1)


def poly_sum_expo(expos):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector of :func:`poly_sum`

    Parameters
    ----------
    expos : tuple of array_like
        Exponent vectors :math:`q, r`

    Returns
    -------
    expo :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    Ms = poly_sum_coef_Ls(expos)
    return np.sum([np.dot(M, expo) for M, expo in zip(Ms, expos)], axis=0)


def poly_sum_expo_Ms(expos):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \dots +\color{blue}{M_2} r}`

    Coefficient Manipulation Matrices of :func:`poly_sum`

    Parameters
    ----------
    expos : tuple of array_like
        Exponent vectors :math:`q, r`

    Returns
    -------
    Ms : list of :class:`~numpy.ndarray`
        Exponent Manipulation Matrices :math:`M_1, M_2,\dots`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.4)
    """
    indices_or_sections = np.cumsum([len(expo) for expo in expos[0:-1]])
    Q = sum(len(expo) for expo in expos)
    return np.split(np.eye(Q), indices_or_sections, axis=1)


def poly_prod(polys):
    r""" :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^q}`

    Product of two univariate polynomials same variable

    Parameters
    ----------
    polys : tuple of :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q, \beta^\mathsf{T} x^q, \dots`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.14)

    """
    assert len(polys) <= 2, "Not yet implemented. Only two polynomials allowed"
    coef = poly_prod_coef(polys)
    expo = poly_prod_expo([poly.expo for poly in polys])
    return lm.Poly(coef, expo)


def poly_prod_coef(polys):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector of :func:`poly_prod`

    Parameters
    ----------
    polys : tuple of :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q, \beta^\mathsf{T} x^q, \dots`

    Returns
    -------
    coef :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.kron(polys[0].coef, polys[1].coef)


def poly_prod_expo_Ms(expos):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{M_1} q + \color{blue}{M_2} r}`

    Exponent Matrices for :class:`poly_prod`

    Parameters
    ----------
    expos : tuple of array_like
        Exponent vectors :math:`q, r`

    Returns
    -------
    Ms : list of :class:`~numpy.ndarray`
        Exponent Manipulation Matrices :math:`M_1, M_2`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.16)

    """
    Q = len(expos[0])
    R = len(expos[1])
    return (
        np.kron(np.identity(Q), np.ones((R, 1))),
        np.kron(np.ones((Q, 1)), np.identity(R)),
    )


def poly_prod_expo(expos):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector of :func:`poly_prod`

    Parameters
    ----------
    expos : tuple of array_like
        Exponent vectors :math:`q, r`

    Returns
    -------
    expo :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.16)

    """
    M1, M2 = poly_prod_expo_Ms(expos)
    return np.add(np.dot(M1, expos[0]), np.dot(M2, expos[1]))


def poly_square(poly):
    r""" :math:`(\alpha^\mathsf{T} x^q)^2 = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Square of a univariate polynomial

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.11)

    """
    return poly_prod((poly, poly))


def poly_square_coef(poly):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector of :func:`poly_square`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`

    Returns
    -------
    coef :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.kron(poly.coef, poly.coef)


def poly_square_expo_M(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{M} q}`

    Exponent manipulation matrix for :func:`poly_square`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    M : :class:`~numpy.ndarray`
        Exponent Manipulation Matrix :math:`M`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.10, Eq. 6.13)

    """
    return np.add(*poly_prod_expo_Ms((expo, expo)))


def poly_square_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector for :func:`poly_square`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    expo :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.12)

    """
    return np.dot(np.add(*poly_prod_expo_Ms((expo, expo))), expo)


def poly_shift(poly, gamma):
    r""" :math:`\alpha^\mathsf{T} (x+ \gamma)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Shift a univariate polynomial by constant value

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`
    gamma : float
        Shift Parameter :math:`\gamma`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.28)

    """
    return lm.Poly(coef=poly_shift_coef(poly, gamma), expo=poly_shift_expo(poly.expo))


def poly_shift_coef(poly, gamma):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector for :func:`poly_shift`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`
    gamma : float
        Shift Parameter :math:`\gamma`

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.29)

    """
    return np.dot(poly_shift_coef_L(poly.expo, gamma), poly.coef)


def poly_shift_coef_L(expo, gamma):
    r""" :math:`\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation matrix L for :func:`poly_shift`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`
    gamma : float
        Shift Parameter :math:`\gamma`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        Coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.32)

    """
    Q = len(expo)
    q_tilde = poly_shift_expo(expo)

    L = np.zeros((len(q_tilde), Q))
    for i, qi in enumerate(expo):
        L[:, i] = comb(qi, q_tilde) * np.power(
            gamma, np.subtract(qi, q_tilde).clip(min=0)
        )
    return L


def poly_shift_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector for :func:`poly_shift`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.30)

    """
    return np.arange(max(expo) + 1)


def poly_dilation(poly, eta):
    r""" :math:`\alpha^\mathsf{T} (\eta x)^q = \color{blue}{\tilde{\alpha}^\mathsf{T} x^q}`

    Dilation of a polynomial by constant value

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`
    eta : float
        Dilation value :math:`\eta`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\tilde{\alpha}^\mathsf{T} x^q`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.33)
    """
    return lm.Poly(coef=poly_dilation_coef(poly, eta), expo=poly.expo)


def poly_dilation_coef(poly, eta):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^q`

    Coefficient vector for :func:`poly_dilation`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`
    eta : float
        Dilation value :math:`\eta`

    Returns
    -------
    coef :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.34)

    """
    return np.dot(poly_dilation_coef_L(poly.expo, eta), poly.coef)


def poly_dilation_coef_L(expo, eta):
    r""" :math:`\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\q}`

    Coefficient manipulation matrix for :func:`poly_dilation`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`
    eta : float
        Dilation value :math:`\eta`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        Coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.35)

    """
    return np.diag(np.power(eta, expo))


def poly_int(poly):
    r""" :math:`\int \big(\alpha^{\mathsf{T}}x^q\big) dx = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Integral of a polynomial

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.17)
    """
    return lm.Poly(poly_int_coef(poly), poly_int_expo(poly.expo))


def poly_int_coef(poly):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector for :func:`poly_int`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.18)
    """
    return mpoly_int_coef(poly, 0)


def poly_int_coef_L(expo):
    r""" :math:`\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation matrix for :func:`poly_int`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        Coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.20-21)
    """
    return mpoly_int_coef_L((expo,), 0)


def poly_int_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector for :func:`poly_int`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.19)
    """
    return mpoly_int_expos((expo,), 0)[0]


def poly_diff(poly):
    r""" :math:`\frac{d}{dx} \big(\alpha^{\mathsf{T}}x^q\big) = \color{blue}{\tilde{\alpha}^\mathsf{T} x^\tilde{q}}`

    Derivative of a polynomial

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\tilde{\alpha}^\mathsf{T} x^\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.24)
    """
    return lm.Poly(poly_diff_coef(poly), poly_diff_expo(poly.expo))


def poly_diff_coef(poly):
    r""" :math:`\color{blue}{\tilde{\alpha}}^\mathsf{T} x^\tilde{q}`

    Coefficient vector for :func:`poly_diff`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        :math:`\alpha^\mathsf{T} x^q`

    Returns
    -------
    coef : :class:`~numpy.ndarray`
        Coefficient vector :math:`\tilde{\alpha}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.25)
    """
    return np.dot(poly_diff_coef_L(poly.expo), poly.coef)


def poly_diff_coef_L(expo):
    r""" :math:`\color{blue}{\Lambda} \alpha^\mathsf{T} x^{\tilde{q}}`

    Coefficient manipulation matrix for :func:`poly_diff`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    L : :class:`~numpy.ndarray`
        Coefficient Manipulation Matrices :math:`\Lambda`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.27)
    """
    return np.diag(expo)


def poly_diff_expo(expo):
    r""" :math:`\tilde{\alpha}^\mathsf{T} x^{\color{blue}{\tilde{q}}}`

    Exponent vector for :func:`poly_diff`

    Parameters
    ----------
    expo : array_like
        Exponent vector :math:`q`

    Returns
    -------
    expo : :class:`~numpy.ndarray`
        Exponent vector :math:`\tilde{q}`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.26)
    """
    return np.subtract(expo, 1).clip(min=0)


def mpoly_add(poly1, poly2):
    """
    Sum of two univariate polynomials different variables

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.37)

    """
    coefs = mpoly_add_coefs(poly1, poly2)
    expos = mpoly_add_expos(poly1, poly2)
    return lm.MPoly(coefs=(np.sum(coefs, axis=0),), expos=expos)


def mpoly_add_coefs(poly1, poly2):
    """
    Coefficients for :func:`mpoly_add`

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`
        Tuple of coefficient vectors

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.38-39)
    """
    Q = poly1.coef_count
    R = poly2.coef_count
    coef1 = np.kron(
        np.concatenate([[0], poly1.coef], axis=0),
        np.concatenate([[1], np.zeros((R,))], axis=0),
    )
    coef2 = np.kron(
        np.concatenate([[1], np.zeros((Q,))], axis=0),
        np.concatenate([[0], poly2.coef], axis=0),
    )
    return coef1, coef2


def mpoly_add_expos(poly1, poly2):
    """
    Exponents for :func:`mpoly_add`

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`
        Tuple of exponent vectors

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.38-39)

    """
    return (
        np.concatenate([[0], poly1.expo], axis=0),
        np.concatenate([[0], poly2.expo], axis=0),
    )


def mpoly_multiply(poly1, poly2):
    """
    Product of two univariate polynomials different variables

    Parameters
    ----------
    poly1 : :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`
    poly2 : :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.40)

    """
    return lm.MPoly(coefs=poly1.coefs + poly2.coefs, expos=poly1.expos + poly2.expos)


def mpoly_prod(polys):
    """
    Product of univariate polynomials different variables

    Parameters
    ----------
    polys : list of :class:`~lmlib.polynomial.poly.Poly` or :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.40)

    """
    coefs = sum(poly.coefs for poly in polys)
    expos = sum(poly.expo for poly in polys)
    return lm.MPoly(coefs, expos)


def mpoly_square(mpoly):
    """
    Square of multivariate polynomial

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.42 - 6.43)

    """
    if len(mpoly.coefs) is 2:

        # implementation factorized polynomial [Wildhaber2019] (Eq. 6.42)
        coef = lm.kron_sequence(
            [mpoly.coefs[0], mpoly.coefs[0], mpoly.coefs[1], mpoly.coefs[1]]
        )
    elif len(mpoly.coefs) is 1:

        # implementation non-factorized polynomial [Wildhaber2019] (Eq. 6.43)
        coef = mpoly_square_coef(mpoly)
    else:
        coef = []
        NotImplementedError("Not yet implemented")

    expos = mpoly_square_expos(mpoly.expos)
    return lm.MPoly(coefs=(coef,), expos=expos)


def mpoly_square_coef(mpoly):
    """
    Non-factorized coefficient vector for :func:`mpoly_square`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.44)

    """
    return np.dot(
        permutation_matrix_square(len(mpoly.expos[0]), len(mpoly.expos[1])),
        np.kron(mpoly.coefs[0], mpoly.coefs[0]),
    )


def mpoly_square_expos(expos):
    """
    Exponent vectors for :func:`mpoly_square`

    Parameters
    ----------
    expos : tuple of array_like

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.45)

    """
    M1, M2 = poly_prod_expo_Ms(expos)
    return (np.dot(np.add(M1, M2), expos[0]),) * 2


def mpoly_shift(poly):
    """
    Polynomial with variable shift

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.49)

    """
    expos = mpoly_shift_expos(poly.expo)
    coefs = (mpoly_shift_coef(poly),)
    return lm.MPoly(coefs, expos)


def mpoly_shift_coef(poly):
    """
    Coefficient vector for :func:`mpoly_shift`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.50)

    """
    return np.dot(mpoly_shift_coef_L(poly.expo), poly.coef)


def mpoly_shift_coef_L(expo):
    """
    Coefficient manipulation matrix for :func:`mpoly_shift`

    Parameters
    ----------
    expo : array_like

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.52-6.53)

    """
    q_tilde = mpoly_shift_expos(expo)[0]
    L = []
    for n in np.arange(max(expo) + 1):
        G = np.zeros((len(q_tilde), len(expo)))
        for i, ri in enumerate(expo):
            for j, sj in enumerate(q_tilde):
                if ri - sj == n:
                    G[j, i] = comb(ri, sj)
        L.append(G)
    return np.concatenate(L, axis=0)


def mpoly_shift_expos(expo):
    """
    Exponent vector for :func:`mpoly_shift`

    Parameters
    ----------
    expo : array_like

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.51)

    """
    return (np.arange(max(expo) + 1),) * 2


def mpoly_int(mpoly, position):
    """
    Integral of a multivariate polynomial with respect to the scalar at a position

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.60 - 6.61)

    """
    coefs = (mpoly_int_coef(mpoly, position),)
    expos = mpoly_int_expos(mpoly.expos, position)
    return lm.MPoly(coefs, expos)


def mpoly_int_coef(mpoly, position):
    """
    Coefficient vector for :func:`mpoly_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61)

    """
    return np.dot(mpoly_int_coef_L(mpoly.expos, position), mpoly.coefs[0])


def mpoly_int_coef_L(expos, position):
    """
    Coefficient manipulation matrix for :func:`mpoly_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61)

    """
    return np.diag(
        lm.kron_sequence(
            [
                np.ones((len(expo),)) if n != position else (1 / (expo + 1))
                for n, expo in enumerate(expos)
            ]
        )
    )


def mpoly_int_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61, Eq.6.19)

    """

    return expos[0:position] + (np.add(expos[position], 1),) + expos[position + 1 : :]


def mpoly_def_int(mpoly, position, a, b):
    """
    Definite integral of a multivariate polynomial with respect to the scalar at a position

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.62 - 6.63)

    """
    coefs = (mpoly_def_int_coef(mpoly, position, a, b),)
    expos = mpoly_def_int_expos(mpoly.expos, position)
    return lm.MPoly(coefs, expos)


def mpoly_def_int_coef(mpoly, position, a, b):
    """
    Coefficient vector for :func:`mpoly_def_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61), (Eq. 6.58)

    """
    return np.dot(mpoly_def_int_coef_L(mpoly.expos, position, a, b), mpoly.coefs[0])


def mpoly_def_int_coef_L(expos, position, a, b):
    """
    Coefficient manipulation matrix for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    a : scalar
        lower integration boundary
    b : scalar
        upper integration boundary

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61), (Eq. 6.58)

    """
    L_int = mpoly_int_coef_L(expos, position)
    expos_int = mpoly_int_expos(expos, position)
    L = lm.kron_sequence(
        [
            np.eye(len(expo))
            if n != position
            else np.atleast_2d(np.power(b, expo) - np.power(a, expo))
            for n, expo in enumerate(expos_int)
        ]
    )
    return np.dot(L, L_int)


def mpoly_def_int_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.61, Eq.6.19)

    """
    return mpoly_substitute_expos(expos, position)


def mpoly_substitute(mpoly, position, substitute):
    """
    Substituting a variable of a multivariate polynomial by a constant

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    substitute : scalar

    Returns
    -------
    out : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.62 - 6.63)

    """
    coefs = (mpoly_substitute_coef(mpoly, position, substitute),)
    expos = mpoly_substitute_expos(mpoly.expos, position)
    return lm.MPoly(coefs, expos)


def mpoly_substitute_coef(mpoly, position, substitute):
    """
    Coefficient vector for :func:`mpoly_def_int`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    substitute : scalar

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return np.dot(
        mpoly_substitute_coef_L(mpoly.expos, position, substitute), mpoly.coefs[0]
    )


def mpoly_substitute_coef_L(expos, position, substitute):
    """
    Coefficient manipulation matrix for :func:`mpoly_def_int`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    substitute : scalar

    Returns
    -------
    L : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return lm.kron_sequence(
        [
            np.eye(len(expo))
            if n != position
            else np.atleast_2d(np.power(substitute, expo))
            for n, expo in enumerate(expos)
        ]
    )


def mpoly_substitute_expos(expos, position):
    """
    Exponent vectors for :func:`mpoly_substitute`

    Parameters
    ----------
    expos : tuple of array_like
    position : int

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.58)

    """
    return tuple(expo for n, expo in enumerate(expos) if n != position)


def mpoly_dilate(mpoly, position, eta):
    """
    Dilate a multivariate polynomial by a constant eta

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    eta: scalar

    Returns
    -------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    coefs = mpoly_dilate_coefs(mpoly, position, eta)
    expos = mpoly_dilate_expos(mpoly.expos)
    return lm.MPoly(coefs, expos)


def mpoly_dilate_coefs(mpoly, position, eta):
    """
    Coefficient vectros for :func:`mpoly_dilate`

    Parameters
    ----------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
    position : int
    eta: scalar

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return (np.dot(mpoly_dilate_coef_L(mpoly.expos, position, eta), mpoly.coefs[0]),)


def mpoly_dilate_coef_L(expos, position, eta):
    """
    Coefficient manipulation matrix for :func:`mpoly_dilate`

    Parameters
    ----------
    expos : tuple of array_like
    position : int
    eta: scalar

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return np.diag(
        lm.kron_sequence(
            [
                np.ones_like(expo) if n != position else np.power(eta, expo)
                for n, expo in enumerate(expos)
            ]
        )
    )


def mpoly_dilate_expos(expos):
    """
    Exponent vectors for :func:`mpoly_dilate`

    Parameters
    ----------
    expos : tuple of array_like

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return expos


def mpoly_dilate_ind(poly):
    r""" :math:`\alpha^{mathsf{T}}(xy)^q = (\Delta_Q\alpha)^{mathsf{T}}(x^q \otimes y^q)`

    Dilates a univariate polynomial `poly` by an indeterminate y

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        Univariate polynomial ``Poly(alpha, q)``

    Returns
    -------
    mpoly : :class:`~lmlib.polynomial.poly.MPoly`
        Multivariate polynomial ``Poly((alpha_tilde,), (q, q))`` with dilation variable :math:`y`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    coefs = mpoly_dilate_ind_coefs(poly)
    expos = mpoly_dilate_ind_expos(poly.expo)
    return lm.MPoly(coefs, expos)


def mpoly_dilate_ind_coefs(poly):
    """
    Coefficient vectros for :func:`mpoly_dilate`

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`

    Returns
    -------
    out : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return (np.dot(mpoly_dilate_ind_coef_L(poly.expo), poly.coef),)


def mpoly_dilate_ind_coef_L(expo):
    """
    Coefficient manipulation matrix for :func:`mpoly_dilate`

    Parameters
    ----------
    expo : tuple of array_like

    Returns
    -------
    out : :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return np.kron(np.eye(len(expo)), np.ones((len(expo), 1))) * np.kron(
        np.atleast_2d(np.eye(len(expo)).flatten("F")).T, np.ones_like(expo).T
    )


def mpoly_dilate_ind_expos(expo):
    """
    Exponent vectors for :func:`mpoly_dilate`

    Parameters
    ----------
    expos : tuple of array_like

    Returns
    -------
    expos : tuple of :class:`~numpy.ndarray`

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.55)

    """
    return expo, expo


def permutation_matrix(m, n, i, j):
    r"""
    Returns permutation matrix

    The permutation is given by

    .. math::

        vec(A\otimes B) = R_{m,n;i,j} \big(vec(A) \otimes vec(B)\big)

    with permutation matrix

    .. math::
        R_{m,n;i,j} = I_n \otimes K_{m,j} \otimes I_i \in \mathbb{R}^{mnij \times mnij}

    and :math:`A_{\{m,n\}} \in \mathbb{R}` and :math:`B_{\{i,j\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    n : int
        Size of second dimension of A
    i : int
        Size of first dimension of B
    j : int
        Size of second dimension of B

    Returns
    -------
    R : :class:`~numpy.ndarray`
        Commutation matrix

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.100-6.102)

    See Also
    --------
    permutation_matrix_square

    """
    K = commutation_matrix(m, j)
    return lm.kron_sequence([np.identity(n), K, np.identity(i)])


def permutation_matrix_square(m, i):
    r"""
    Returns permutation matrix for square matrices A and B

    The permutation is given by

    .. math::

        vec(A\otimes B) = R_{m,n;i,j} \big(vec(A) \otimes vec(B)\big)

    with permutation matrix

    .. math::

        R_{m;i} = R_{m,m;i,i}

    and :math:`A_{\{m,m\}} \in \mathbb{R}` and :math:`B_{\{i,i\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    i : int
        Size of first dimension of B

    Returns
    -------
    R : :class:`~numpy.ndarray`
        Commutation matrix

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.101-6.103)

    See Also
    --------
    permutation_matrix

    """
    return permutation_matrix(m, m, i, i)


def commutation_matrix(m, n):
    r"""
    Returns commutation matrix

    .. math::

        K_{m,n}vec(A) = vec(A^\mathsf{T}) \in \mathbb{R}^{mn \times mn}

    where :math:`A_{\{m,n\}} \in \mathbb{R}`

    Parameters
    ----------
    m : int
        Size of first dimension of A
    n : int
        Size of second dimension of A

    Returns
    -------
    K : :class:`~numpy.ndarray`
        Commutation matrix squared

    References
    ----------
    [Wildhaber2019]_ (Eq. 6.114-6.115)

    """
    K = np.zeros((m * n, m * n))
    for i in range(m):
        em = np.zeros((m,))
        em[i] = 1
        for j in range(n):
            en = np.zeros((n,))
            en[j] = 1
            K += np.kron(np.outer(em, en), np.outer(en, em))
    return K
