r"""
This module provides a calculus for uni- and multivariate polynomials using the vector exponent notation [Wildhaber2019]_, see Chapter 6,
with the intention to use polynomials in (squared error) cost functions e.g., as localized signal models.


This module supports polynomials of the following forms:

* **univariate polynomial in** :math:`x \in \mathbb{R}`

    :math:`\alpha^{\mathsf{T}}x^q \in \mathbb{R}`

    with coefficient vector :math:`\alpha \in \mathbb{R}^{Q}`,
    and with exponent vectors
    :math:`q \in \mathbb{N}_{0}^{Q}`.


    **Example:**

    .. math::

      \alpha^{\mathsf{T}} x^q = p(x) =a_0 x^{q_0} + a_1 x^{q_1} + \dots + a_{Q-1} x^{q_{Q-1}}


    where :math:`\alpha^{\mathsf{T}} = [a_0, a_1, \dots, a_{Q-1}]` and :math:`q = [q_0, q_1, \dots, q_{Q-1}]^{\mathsf{T}}`.
    [Wildhaber2019]_ [Eq 6.1].


* **non-factorized multivariate polynomial in** :math:`x, y \in \mathbb{R}`:

    :math:`\tilde{\alpha}^{\mathsf{T}}(x^q \otimes y^r) \in \mathbb{R}`

    with coefficient vector :math:`\tilde{\alpha} \in \mathbb{R}^{QR}`,
    and with exponent vectors
    :math:`q \in \mathbb{N}_{0}^{Q}` and
    :math:`r \in \mathbb{N}_{0}^{R}`.

* **factorized multivariate polynomial in** :math:`x, y \in \mathbb{R}`:

    :math:`(\alpha \otimes \beta)^{\mathsf{T}}(x^q \otimes y^r) = (\alpha^{\mathsf{T}}x^q) (\beta^{\mathsf{T}} y^r)  \in \mathbb{R}`

    all with coefficient vector
    :math:`\alpha \in \mathbb{R}^{Q}` and
    :math:`\beta \in \mathbb{R}^{R}`,
    and with exponent vectors
    :math:`q \in \mathbb{N}_{0}^{Q}` and
    :math:`r \in \mathbb{N}_{0}^{R}`.


This module contains:
*********************
- Class :class:`~lmlib.polynomial.poly.Poly` : base class of all univariate polynomial in vector exponent notation
- Class :class:`~lmlib.polynomial.poly.MPoly` : base class of all multivariate polynomial in vector exponent notation
- Functions :func:`poly_<name>()`: manipulators for univariate polynomials
- Functions :func:`mpoly_<name>()`: manipulators for multivariate polynomials




Polynomial Base Classes
***********************

.. currentmodule:: lmlib.polynomial.poly

.. autosummary::
   :toctree: _polynomial

   Poly
   MPoly


Notes
*****
The class :class:`~lmlib.polynomial.poly.Poly` is inherited from :class:`~lmlib.polynomial.poly.MPoly`.
They differ mainly in the interface and evaluation.



Polynomial Operations
*********************
.. currentmodule:: lmlib.polynomial.operations


.. contents::
   :local:


Miscellaneous Operations
------------------------

.. autosummary::
   :toctree: _polynomial

   permutation_matrix
   permutation_matrix_square
   commutation_matrix

Operations with Univariate Polynomial Output
--------------------------------------------
The **output** of the following operator functions concerns just **univariate** polynomials,
even the input arguments involve multiple polynomials.

Sum of Polynomials :math:`\alpha^\mathsf{T} x^q + \beta^\mathsf{T} x^r`
.......................................................................

.. autosummary::
   :toctree: _polynomial

   poly_sum
   poly_sum_coef
   poly_sum_coef_Ls
   poly_sum_expo
   poly_sum_expo_Ms


.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>> p2 = lm.Poly([2, -1], [0, 1])
   >>>
   >>> p_sum = lm.poly_sum((p1, p2))
   >>> print(p_sum)
   [ 1.  3.  5.  2. -1.], [0. 1. 2. 0. 1.]


Product of Polynomials :math:`\alpha^\mathsf{T} x^q \cdot \beta^\mathsf{T} x^r`
...............................................................................

.. autosummary::
   :toctree: _polynomial

   poly_prod
   poly_prod_coef
   poly_prod_expo
   poly_prod_expo_Ms


.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>> p2 = lm.Poly([2, -1], [0, 1])
   >>>
   >>> p_prod = lm.poly_prod((p1, p2))
   >>> print(p_prod)
   [ 2 -1  6 -3 10 -5], [0. 1. 1. 2. 2. 3.]


Square of a Polynomial :math:`(\alpha^\mathsf{T} x^q)^2`
........................................................

.. autosummary::
   :toctree: _polynomial

   poly_square
   poly_square_coef
   poly_square_expo
   poly_square_expo_M


.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>>
   >>> p_square = lm.poly_square(p1)
   >>> print(p_square)
   [ 1  3  5  3  9 15  5 15 25], [0. 1. 2. 1. 2. 3. 2. 3. 4.]


Shift of a Polynomial :math:`\alpha^\mathsf{T} (x+ \gamma)^q`
...........................................................

.. autosummary::
   :toctree: _polynomial

   poly_shift
   poly_shift_coef
   poly_shift_coef_L
   poly_shift_expo

.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>> gamma = 2
   >>> p_shift = lm.poly_shift(p1, gamma)
   >>> print(p_shift)
   [27. 23.  5.], [0 1 2]


Dilation of a Polynomial :math:`\alpha^\mathsf{T} (\eta x)^q`
................................................................

.. autosummary::
   :toctree: _polynomial

   poly_dilation
   poly_dilation_coef
   poly_dilation_coef_L

.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>> eta = -5
   >>> p_dilation = lm.poly_dilation (p1, eta)
   >>> print(p_dilation)
   [  1 -15 125], [0 1 2]


Integral of a Polynomial :math:`\int (\alpha^\mathsf{T} x^q) dx`
................................................................

.. autosummary::
   :toctree: _polynomial

   poly_int
   poly_int_coef
   poly_int_coef_L
   poly_int_expo

.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>>
   >>> p_int = lm.poly_int(p1)
   >>> print(p_int)
   [1.         1.5        1.66666667], [1 2 3]



Derivative of a Polynomial :math:`\frac{d}{dx} (\alpha^\mathsf{T} x^q)`
.......................................................................

.. autosummary::
   :toctree: _polynomial

   poly_diff
   poly_diff_coef
   poly_diff_coef_L
   poly_diff_expo

.. code::

   >>> import lmlib as lm
   >>>
   >>> p1 = lm.Poly([1, 3, 5], [0, 1, 2])
   >>>
   >>> p_diff = lm.poly_diff(p1)
   >>> print(p_diff)
   [ 0  3 10], [0 0 1]



Multivariate Polynomial Output
------------------------------

.. autosummary::
   :toctree: _polynomial

   mpoly_add
   mpoly_add_coefs
   mpoly_add_expos
   mpoly_multiply
   mpoly_prod
   mpoly_square


"""
from __future__ import division, absolute_import, print_function

from lmlib.polynomial.poly import *
from lmlib.polynomial.operations import *
from lmlib.polynomial.solver import *
