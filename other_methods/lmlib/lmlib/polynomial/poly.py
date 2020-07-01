# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto

from __future__ import division, absolute_import, print_function

__all__ = ["MPoly", "Poly"]

import numpy as np


class MPoly:
    r""" :math:`\big(\alpha \otimes \beta \big)^\mathsf{T} (x^q \otimes y^r)` and :math:`\big(\tilde{\alpha})^\mathsf{T} (x^q \otimes y^r)`

    Polynomial class for multivariate polynomials in vector exponent notation

    Such a multivariate polynomial is given by

    .. math::

        p(x) = (\alpha \otimes \beta)^\mathsf{T}(x^q \otimes y^r) \ ,

    where :math:`\alpha \in \mathbb{R}^Q` and :math:`\beta \in \mathbb{R}^R` are the coefficient vectors,
    :math:`q \in \mathbb{Z}_{\geq 0}^Q` and :math:`r \in \mathbb{Z}_{\geq 0}^R` is the exponent vectors,
    and :math:`x \in \mathbb{R}` and :math:`y \in \mathbb{R}` the independent variables.

    """

    def __init__(self, coefs, expos):
        """
        Parameters
        ----------
        coefs : tuple of array_like
            Set of coefficient vector(s)
        expos : tuple of array_like
            Set of exponent vector(s)
        """

        self.expos = expos
        self.coefs = coefs
        self._check_dimensions()
        self._factorize()

    def __str__(self):
        return "{}, {}".format(self.coefs, self.expos)

    @property
    def expos(self):
        """tuple of :class:`~numpy.ndarray` : Exponent vectors"""
        return self._expos

    @expos.setter
    def expos(self, expos):
        assert isinstance(expos, tuple), "expos is not an instance of tuple"
        self._expos = tuple(np.asarray(expo) for expo in expos)
        self.var_count = len(expos)

    @property
    def var_count(self):
        """int : Number of dependent variables"""
        return self._var_count

    @var_count.setter
    def var_count(self, var_count):
        assert isinstance(var_count, int), "var_count is not an instance of int"
        self._var_count = var_count

    @property
    def coefs(self):
        """tuple of :class:`~numpy.ndarray` : Coefficient vectors"""
        return self._coefs

    @coefs.setter
    def coefs(self, coefs):
        assert isinstance(coefs, tuple), "coefs is not an instance of tuple"
        self._coefs = tuple(np.asarray(coef) for coef in coefs)

    @property
    def coef_fac(self):
        """:class:`~numpy.ndarray` : Factorized coefficient vector"""
        return self._coef_fac

    def _factorize(self):
        kron_coef = 1
        for coef in self._coefs:
            kron_coef = np.kron(kron_coef, coef)
        self._coef_fac = kron_coef

    def _check_dimensions(self):
        nof_coef_elements = np.prod([len(coef) for coef in self._coefs])
        nof_expo_elements = np.prod([len(expo) for expo in self._expos])
        assert nof_coef_elements == nof_expo_elements, (
            "Elements count of the coefficient vectors doesn't coincide "
            "with elements count of the exponent vectors."
        )

    def _eval_scalar(self, variables):

        kron_var_expo = 1
        for variable, expo in zip(variables, self._expos):
            kron_var_expo = np.kron(kron_var_expo, np.power(variable, expo))

        return np.dot(self._coef_fac, kron_var_expo)

    def scale(self, value):
        """Scales internal polynomial coefficients elementwise with `value`

        Parameters
        ==========

        value : scalar
            Scale value
        """
        self.coefs = tuple(np.dot(value, coef) for coef in self.coefs)
        self._factorize()

    def eval(self, variables):
        """Evaluates the multivariat polynomial

        Parameters
        ----------
        variables : tuple of array_like or tuple of scalar

        Returns
        -------
        out : :class:`~numpy.ndarray` or :class:`~numpy.number`

        """
        assert len(variables) == self._var_count, (
            "length of variables doesn't match with var_count "
            "\n Expect: {} \n Found: {}".format(self._var_count, len(variables))
        )

        variables = np.asarray(variables)
        eval_shape = variables[0].shape

        assert all(
            eval_shape == np.shape(variable) for variable in variables
        ), "independent variables has different number of elements"

        out = np.empty(eval_shape)
        it = np.nditer(out, flags=["multi_index"])
        while not it.finished:
            out[it.multi_index] = self._eval_scalar(
                [variable[it.multi_index] for variable in variables]
            )
            it.iternext()
        return out


class Poly(MPoly):
    r""" :math:`\alpha^\mathsf{T} x^q`

    Polynomial class for univariat polynomials in vector exponent notation

    Such a polynomial is given by

    .. math::

        p(x) &= \alpha^\mathsf{T}x^q\\
             &= \begin{bmatrix}a_0& a_1& \cdots& a_{Q-1}\end{bmatrix}
             \begin{bmatrix}x^{q_0}\\ x^{q_1}\\ \vdots\\ x^{q_{Q-1}}\end{bmatrix}\\
             &= a_0 x^{q_0} + a_1 x^{q_1}+ \dots + a_{Q-1} x^{q_{Q-1}} \ ,

    where the :math:`\alpha \in \mathbb{R}^Q` is the coefficient vector,
    :math:`q \in \mathbb{Z}_{\geq 0}^Q` is the exponent vector,
    and :math:`x \in \mathbb{R}` the independent variable.
    """

    def __init__(self, coef, expo):
        """
        Parameters
        ----------
        coef : array_like
            shape=(`Q`),
            Coefficient vector
        expo : array_like
            shape=(`Q`),
            Exponent vector


        ---

        | |def_poly_Q|
        """
        self.expo = np.asarray(expo)
        self.coef = np.asarray(coef)
        super(Poly, self).__init__((self._coef,), (self._expo,))

    def __str__(self):
        return "{}, {}".format(self.coef, self.expo)

    @property
    def expo(self):
        """:class:`~numpy.ndarray` : Exponent vector :math:`q`"""
        return self._expo

    @expo.setter
    def expo(self, expo):
        self._expo = expo

    @property
    def coef(self):
        """:class:`~numpy.ndarray` : Coefficient vector :math:`\\alpha`"""
        return self._coef

    @coef.setter
    def coef(self, coef):
        self._coef = coef

    @property
    def coef_count(self):
        return len(self._coef)

    def eval(self, variable):
        """
        Evaluates the univariat polynomial

        Parameters
        ----------
        variable : array_like or scalar
            arbitrary shape,


        Returns
        -------
        out : :class:`~numpy.ndarray` or :class:`~numpy.number`
            shape(`variable`),

        """
        assert not isinstance(
            variable, tuple
        ), "variable is instance tuple, expect array_like or scalar"
        return super(Poly, self).eval((variable,))
