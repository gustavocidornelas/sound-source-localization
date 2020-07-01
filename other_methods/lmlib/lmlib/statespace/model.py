# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
"""
Various Linear State Space Models

Autonomous Linear State Space Models (ALSSMs)
---------------------------------------------
.. autosummary::
   :toctree: _model/_classes

   Alssm
   AlssmPoly
   AlssmExp
   AlssmSin

Modifier/Container Classes for ALSSMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The abstract base class for such ALSSMs is :class:`AlssmContainer`.

.. autosummary::
   :toctree: _model/_classes

   AlssmContainer
   AlssmStacked
   AlssmStackedSO
   AlssmProd


Linear State Space Models (LSSMs)
---------------------------------

.. autosummary::
   :toctree: _model/_classes

   Lssm
   LssmPoly
   LssmExp
   LssmSin

Modifier/Container Classes for LSSMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The abstract base class for such LSSMs is :class:`LssmContainer`.

.. autosummary::
   :toctree: _model/_classes

   LssmContainer
   LssmStacked
   LssmStackedSO
   LssmStackedCI
   LssmStackedCISO

"""

from __future__ import division, absolute_import, print_function

__all__ = [
    "Alssm",
    "AlssmPoly",
    "AlssmSin",
    "AlssmExp",
    "AlssmContainer",
    "AlssmStacked",
    "AlssmStackedSO",
    "AlssmProd",
    "Lssm",
    "LssmPoly",
    "LssmSin",
    "LssmExp",
    "LssmContainer",
    "LssmStacked",
    "LssmStackedSO",
    "LssmStackedCI",
    "LssmStackedCISO",
]

import numpy as np
from scipy.linalg import pascal, block_diag
from abc import ABC, abstractmethod

import lmlib as lm


# Base Classes


class _ModelBase(ABC):
    """Abrstract baseclass for (autonomous) linear state space models.

    Note
    ----
    This class has the abstract method :meth:`update`. A class that are derived from :class:`_ModelBase` cannot be
    instantiated unless all of its abstract methods and abstract properties are overridden.

    """

    _isUpdated = False
    _A = np.empty((0, 0))
    _C = np.empty((0, 0))
    _label = "label missing"

    @property
    def isUpdated(self):
        """bool : Returns True if the model has updated it latest changes."""
        return self._isUpdated

    @isUpdated.setter
    def isUpdated(self, isUpdated):
        self._isUpdated = isUpdated

    @property
    def A(self):
        """:class:`~numpy.ndarray` : State transition matrix :math:`A \\in \\mathbb{R}^{N \\times N}`"""
        return self._A

    @A.setter
    def A(self, A):
        self._A = A
        self.isUpdated = False

    @property
    def C(self):
        """:class:`~numpy.ndarray` : Output matrix :math:`C \\in \\mathbb{R}^{L \\times N}`"""
        return self._C

    @C.setter
    def C(self, C):
        self._C = C
        self.isUpdated = False

    @property
    def label(self):
        """str : Label of the model"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def N(self):
        """int : Order :math:`N`"""
        return self.A.shape[0] if self.A.ndim is 2 else self.A.shape

    @property
    def output_count(self):
        """int : Number of outputs :math:`L`"""
        if self.C.ndim is 0:
            return 0
        elif self.C.ndim is 1:
            return 1
        else:
            return self.C.shape[0]

    @property
    def hasVectorOutput(self):
        """bool : Returns True if the output of the model has at least one dimension."""
        return True if self.C.ndim is 2 else False

    @abstractmethod
    def update(self):
        pass

    def dump_tree(self, level=0):
        """
        Returns the structure of the model as a string.

        Parameters
        ----------
        level : int
            String-indent level (used for recursive calls in nested LSSMs)

        Returns
        -------
        out : str
            String representing internal model structure.

        Examples
        --------

        - :ref:`lmlib_example_stacked_linear_state_space_models`
        - :ref:`lmlib_example_cost_segment`

        >>> alssm = lm.AlssmPoly(Q=3)
        >>> alssm.update()
        >>> str = alssm.dump_tree()
        >>> print(str)
        └-Alssm : polynomial, A:(4, 4), C:(4,), label: label missing
        """
        return ("  " * level + "└-" + self.__repr__()) + "\n"


class _ModelContainer(ABC):
    """Abrstract containter class"""

    _models = ()

    @abstractmethod
    def __init__(self, models, G=None):
        self.G = G

    @property
    def G(self):
        """list, tuple, None : Scaling factors for the output matrix of the ALSSM in `alssms`."""
        return self._G

    @G.setter
    def G(self, G):
        self._G = G
        self.isUpdated = False

    def dump_tree(self, level=0):
        """
        Returns the internal structure of LSSMs as a string.

        Parameters
        ----------
        level : int
            String-indent level (used for recursive calls in nested LSSMs)

        Returns
        -------
        out : str
            Dump String representing internal LSSM structure.

        Examples
        --------
        >>> import lmlib as lm
        >>>
        >>> poly = lm.AlssmPoly(4, label="polynomial model")
        >>>
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> line = lm.Alssm(A, C, label="line model")
        >>>
        >>> stacked_alssm = lm.AlssmStacked((poly, line), label='stacked model')
        >>> stacked_alssm.update()
        >>> print(stacked_alssm.dump_tree())

        """
        str_tree = ("  " * level + "└-" + self.__repr__()) + "\n"
        for model in self._models:
            str_tree += model.dump_tree(level=level + 1)
        return str_tree


# Autonomous linear state space models


class Alssm(_ModelBase):
    r"""
    Autonomous Linear State Space Model (ALSSM) class

    This class holds the parameters of a discrete-time, autonomous (i.e., input-free), single- or multi-output linear
    state space model, defined recursively by

    .. math::
       x[k+1] &= Ax[k]

       y[k] &= Cx[k],

    where :math:`A \in \mathbb{R}^{N\times N}, C \in \mathbb{R}^{L \times N}` are the fixed model parameters (matrices),
    :math:`k` the time index,
    :math:`y` the output signal,
    and :math:`x` the state vector.

    For more details, see also [Wildhaber2019]_ [Eq. 4.1].

    Notes
    -----
    |note_MC|

    """

    alssm_type_str = "native"
    """str: String representing the current class type (in a well-readable form)"""

    def __init__(self, A, C, label="label missing"):
        r"""
        Parameters
        ----------
        A : array_like
            shape=(`N`, `N`),
            Sets the state transition matrix :math:`A \in \mathbb{R}^{N \times N}`.
        C : array_like
            shape=([`L`,] `N`),
            Sets the output matrix :math:`C \in \mathbb{R}^{L \times N}`.
            See :ref:`single_multi_output_description`.
        label : str
            Sets the model label.


        ---

        | |def_N|
        | |def_L|


        See Also
        --------
        Lssm

        Examples
        --------
        Setting up an autonomous linear state space model.

        >>> import lmlib as lm
        >>>
        >>> A = [[1, 1, 1], [0, 1, 2], [0, 0, 1]]
        >>> C = [1, 0, 0]
        >>> alssm = lm.Alssm(A, C)
        >>> alssm.update()
        >>> print(alssm)
        A =
        [[1 1 1]
         [0 1 2]
         [0 0 1]]
        C =
        [1 0 0]

        """
        self.A = np.asarray(A)
        self.C = np.asarray(C)
        self.label = label
        self.update()

    def __repr__(self):
        return (
            "Alssm : "
            + self.alssm_type_str
            + ", A:"
            + str(self.A.shape)
            + ", C:"
            + str(self.C.shape)
            + ", label: "
            + self.label
        )

    def __str__(self):
        return "A = \n{}\nC = \n{}".format(self.A.__str__(), self.C.__str__())

    def _check_dimensions(self):
        assert self.A.shape[0] == self.A.shape[0], (
            "State Transition matrix is not square."
            + "\n"
            + "Found A shape: "
            + str(self.A.shape)
        )
        assert self.C.shape[-1] == self.A.shape[0], (
            "Last dimension of output matrix doesn't match with system order."
            + "\n"
            + self.__str__()
            + "\n"
            + "Found A shape: "
            + str(self.A.shape)
            + "\n"
            + "Found C shape: "
            + str(self.C.shape)
        )

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        self._check_dimensions()
        self.isUpdated = True

    def eval_at(self, xs, js, ref=()):
        r"""Evaluates the ALSSM with the state space vector x at the discrete time index j

        .. math::
            s_j(x)  = CA^jx

        Parameters
        ----------
        xs : array_like
            shape=(`K`,`N`,[`S`]),
            List of `K` initial state space vector :math:`x \in \mathbb{R}^N`
        js : list of int
            Lost of discrete time indices :math:`j \in \mathbb{Z}`
        ref : tuple
            TODO Creates a reference... . The number of xs (first dimension) matches with `ks`.
            The first variable in the tuple `ref` is a list of indices `ks` and the second the number of samples `K`.
            For example, ``ref=(ks, K)`` with ``ks = [1,3 7]``, ``K = 10``.

        Returns
        -------
        s : :class:`~numpy.ndarray`
               shape=(`K`,`J`,[`L`,][`S`,]),
               ALSSM output :math:`s_j(x)`
               Missing values in a sparse referenced output are assigned with ``np.nan``.


        ---

        | |def_N|
        | |def_L|
        | |def_S|

        Examples
        --------
        >>> # Single-Output ALSSM
        >>> A = [[1, 1, 1], [0, 1, 2], [0, 0, 1]]
        >>> C = [1, 0, 0]
        >>> alssm = lm.Alssm(A, C)
        >>> alssm.update()
        >>> x = [1, 0, 0]
        >>> s = alssm.eval_at(x, j=0)
        >>> print(s)
        1

        """
        if len(ref) == 0:
            return np.asarray(
                [
                    [(self.C @ np.linalg.matrix_power(self.A, j)) @ x for j in js]
                    for x in xs
                ]
            )

        ks = ref[0]
        K = ref[1]
        J = len(js)
        L_shape = (self.output_count,) if self.hasVectorOutput else ()
        S_shape = (len(xs[0][0]),) if np.ndim(xs[0]) is 2 else ()
        s = np.full((K, J,) + L_shape + S_shape, np.nan)

        for x, k, in zip(xs, ks):
            s[ks] = [(self.C @ np.linalg.matrix_power(self.A, j)) @ x for j in js]
        return s

    def eval(self, xs, ref=()):
        r"""Evaluates an ALSSM for each state space vector in the list x.

        .. math::
            s_0(x_k) = CA^0x_k = Cx_k

        Parameters
        ----------
        xs : array_like
            shape=(`K`,`N`[, `S`]),
            Array of state space vectors

        ref : tuple
            TODO

        Returns
        -------
        s : :class:`~numpy.ndarray`
            shape=(`len(xs)`, [`L`, [`S`]]),
            Array of ALSSM outputs :math:`s_0(x)=\{s_0(x_k) \mid x_k \in x \}`


        ---

        | |def_N|
        | |def_L|
        | |def_S|

        """
        if len(ref) is 0:
            if self.hasVectorOutput:
                return np.einsum("li, hi...->hl...", self.C, xs)
            return np.einsum("...i, hi...->h...", self.C, xs)

        ks = ref[0]
        K = ref[1]
        L_shape = (self.output_count,) if self.hasVectorOutput else ()
        S_shape = (len(xs[0][0]),) if np.ndim(xs[0]) is 2 else ()
        s = np.full((K,) + L_shape + S_shape, np.nan)

        for x, k, in zip(xs, ks):
            s[k] = self.C @ x
        return s


class AlssmPoly(Alssm):
    r"""Autonomous linear state space model with an impulse response of a polynomial sequence

   Discrete-time linear state space model generating output sequences of shape of a polynomial of order `Q`.

   **Basis**

   * natural form:

       .. math::

           M = \begin{bmatrix}
                   1 & 1 & 1 \\
                   0 & 1 & 2 \\
                   0 & 0 & 1
               \end{bmatrix}

   * diagonal form:

       .. math::

           M = \begin{bmatrix}
                   1 & 1 & 1 \\
                   0 & 1 & 2 \\
                   0 & 0 & 1
               \end{bmatrix}

   """
    alssm_type_str = "polynomial"
    """str: String representing the current class type (in a well-readable form)"""

    def __init__(self, Q, C=None, label="label missing"):
        """

        Parameters
        ----------
        Q : int
            Sets the polynomial order (polynomial degree -1). The ALSSM order `N` is ``Q+1``.
        C : array_like, None
            Sets output matrix. If C is one dimensional of shape ``(N,)`` its a **single-output** ALSSM.
            Two-dimensional output matrix of shape ``(L, N)`` leads to a **multi-output** ALSSM.
            If `C=None` (default) the output matrix gets initialized as single-output ALSSM.
        label : str
            Sets the label of the ALSSM.


        ---

        | |def_N|
        | |def_L|

        See Also
        --------
        Alssm, LssmPoly

        Examples
        --------
        Setting up a 4. order polynomial, autonomous linear state space model.

        >>> import lmlib as lm
        >>>
        >>> Q = 4
        >>> alssm = lm.AlssmPoly(Q)
        >>> alssm.update()
        >>> print(alssm)
        A =
        [[1 1 1 1 1]
         [0 1 2 3 4]
         [0 0 1 3 6]
         [0 0 0 1 4]
         [0 0 0 0 1]]
        C =
        [1. 0. 0. 0. 0.]

        """
        assert Q >= 0 and isinstance(
            Q, int
        ), "Polynomial order has to be a positive integer."
        self.Q = Q
        if C is None:
            C = np.concatenate([[1], np.zeros((Q,))], axis=-1)
        A = np.zeros((Q + 1, Q + 1))
        super(AlssmPoly, self).__init__(A, C, label=label)
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        self.A = pascal(self.Q + 1, kind="upper")
        self._check_dimensions()
        self.isUpdated = True

    @property
    def Q(self):
        """int : Polynomial order :math:`Q`"""
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self.isUpdated = False


class AlssmSin(Alssm):
    r"""Autonomous linear state space model with an impulse response of a (damped) sinusoidal sequence.

    Discrete-time linear state space model generating output sequences of sinusoidal shape with decay factor `rho` and
    discrete-time frequency `omega`.

    .. math::

        A = \begin{bmatrix}
            \rho \cos{\omega} & -\rho \sin{\omega} \\
            \rho \sin{\omega} & \rho \cos{\omega}
           \end{bmatrix}

    """
    alssm_type_str = "sinusoidal"
    """str: String representing the current class type (in a well-readable form)"""

    def __init__(self, omega, rho, C=None, label="label missing"):
        """
        Parameters
        ----------
        omega : float
            Sets the frequency :math:`\\omega = 2\\pi f_s`
        rho : float
            Sets the decay factor
        C : array_like, None
            Sets output matrix. If C is one dimensional of shape ``(N,)`` its a **single-output** ALSSM.
            Two-dimensional output matrix of shape ``(L, N)`` leads to a **multi-output** ALSSM.
            If `C=None` (default) the output matrix gets initialized as single-output ALSSM.
        label : str
            Sets the label of the ALSSM.


        ---

        | |def_N|
        | |def_L|

        See Also
        --------
        Alssm, LssmSin

        Examples
        --------
        Setting up a sinusoidal, autonomous linear state space model.

        >>> import lmlib as lm
        >>>
        >>> omega=0.1
        >>> rho=0.9
        >>> alssm = lm.AlssmSin(omega, rho)
        >>> alssm.update()
        >>> print(alssm)
        A =
        [[ 0.89550375 -0.08985007]
         [ 0.08985007  0.89550375]]
        C =
        [1 0]

        """
        self.omega = omega
        self.rho = rho
        if C is None:
            C = np.array([1, 0])
        A = np.zeros((2, 2))
        super(AlssmSin, self).__init__(A, C, label=label)
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        c, s = np.cos(self.omega), np.sin(self.omega)
        self.A = self.rho * np.array([[c, -s], [s, c]])
        self._check_dimensions()
        self.isUpdated = True

    @property
    def omega(self):
        """float : frequency factor :math:`\\omega = 2\\pi f_s`"""
        return self._omega

    @omega.setter
    def omega(self, omega):
        self._omega = omega
        self.isUpdated = False

    @property
    def rho(self):
        """float : Decay factor :math:`\\rho`"""
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.isUpdated = False


class AlssmExp(Alssm):
    """Exponential autonomous linear state space model

    Discrete-time linear state space model generating output sequences of exponentially decaying shape with decay
    factor :math:`\\gamma`.
    """

    alssm_type_str = "exponential"
    """str: String representing the current class type (in a well-readable form)"""

    def __init__(self, gamma, label="label missing"):
        """
        Parameters
        ----------
        gamma : float
            Sets decay factor per sample ( < 1.0 : left-side decaying; > 1.0 : right-side decaying)
        label : str
            Sets the label of the ALSSM.

        See Also
        --------
        Alssm, LssmExp

        Examples
        --------
        Setting up an exponential, autonomous linear state space model.

        >>> import lmlib as lm
        >>>
        >>> gamma=0.8
        >>> alssm = lm.AlssmExp(gamma)
        >>> alssm.update()
        >>> print(alssm)
        A =
        [[0.8]]
        C =
        [1.]

        """
        self.gamma = gamma
        C = np.ones((1,))
        A = np.zeros((1, 1))
        super(AlssmExp, self).__init__(A, C, label=label)
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        self.A = np.array([[self.gamma]])
        self._check_dimensions()
        self.isUpdated = True

    @property
    def gamma(self):
        """float :  decay factor per sample :math:`\\gamma`"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self.isUpdated = False


# ALSSM modifiers


class AlssmContainer(Alssm, _ModelContainer, ABC):
    """Abstract class for ALSSM Container"""

    @abstractmethod
    def __init__(self, alssms, G, A, C, label):
        """
        Parameters
        ----------
        alssms : tuple of :class:`Alssm`
            shape=(`M`)
            Set of ALSSMs
        G : list
            shape=(`M`)
        A : array_like
            Sets state transition matrix :math:`A`
        C : array_like
            Sets output matrix :math:`C`
        label : str
            Label of the ALSSM


        ---

        | |def_M_indep|
        """
        self.alssms = alssms
        self.G = G
        super(AlssmContainer, self).__init__(A, C, label=label)

    @property
    def alssms(self):
        """tuple : Set of autonomous linear state space models"""
        return self._models

    @alssms.setter
    def alssms(self, alssms):
        for i, alssm in enumerate(alssms):
            assert isinstance(alssm, Alssm), (
                "The object at position "
                + str(i)
                + "in alssms is not of type Alssm."
                + "\n"
                + "Found element of type: "
                + alssm.__name__
            )
        self._models = alssms
        self.isUpdated = False


class AlssmStacked(AlssmContainer):
    """
    Stacking Autonomous linear state space models to one single ALSSM

    The outputs staying individual.
    """

    alssm_type_str = "stacked"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, alssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        alssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        N = np.sum([alssm.N for alssm in self.alssms], dtype=int)
        L = np.sum([alssm.output_count for alssm in self.alssms], dtype=int)
        super(AlssmStacked, self).__init__(
            alssms, G, np.zeros((N, N)), np.zeros((L, N)), label
        )
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        for alssm in self.alssms:
            alssm.update()

        self.A = block_diag(*[alssm.A for alssm in self.alssms])
        if self.G is not None:
            self.C = block_diag(*[g * alssm.C for g, alssm in zip(self.G, self.alssms)])
        else:
            self.C = block_diag(*[alssm.C for alssm in self.alssms])
        self._check_dimensions()
        self.isUpdated = True


class AlssmStackedSO(AlssmContainer):
    """
    Stacking Autonomous linear state space models to one single ALSSM where the outputs are summed.
    """

    alssm_type_str = "stacked-so"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, alssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        alssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        N = 0
        L = 0
        isMO = False
        for alssm in alssms:
            N += alssm.N
            L += alssm.output_count
            isMO = isMO or alssm.hasVectorOutput
        C = np.zeros((L, N)) if isMO else np.zeros((N,))
        super(AlssmStackedSO, self).__init__(alssms, G, np.zeros((N, N)), C, label)
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        isMO = False
        for alssm in self.alssms:
            alssm.update()
            isMO = isMO or alssm.hasVectorOutput

        self.A = block_diag(*[alssm.A for alssm in self.alssms])
        if isMO:
            if self.G is not None:
                self.C = np.concatenate(
                    [
                        g * np.atleast_2d(alssm.C)
                        for g, alssm in zip(self.G, self.alssms)
                    ],
                    axis=-1,
                )
            else:
                self.C = np.concatenate(
                    [np.atleast_2d(alssm.C) for alssm in self.alssms], axis=-1
                )
        else:
            if self.G is not None:
                self.C = np.concatenate(
                    [g * alssm.C for g, alssm in zip(self.G, self.alssms)], axis=-1
                )
            else:
                self.C = np.concatenate([alssm.C for alssm in self.alssms], axis=-1)
        self._check_dimensions()
        self.isUpdated = True


# ALSSMs Product


class AlssmProd(AlssmContainer):
    r"""Product of autonomous linear state space models

    As written in [Wildhaber2019]_ [Eq. 4.21], the multiplication of two ALSSMs is given by

    .. math::

        s_j^{(1)} \cdot s_j^{(2)} &= (C_1 A_1^j x_1)(C_2 A_2^j x_2)\\
        &= (C_1 A_1^j x_1) \otimes (C_2 A_2^j x_2)\\
        &= (C_1 \otimes C_2) (A_1^j \otimes A_2^j) (x_1 \otimes  x_2) \ ,

    where :math:`s_j^{(1)} = C_1 A_1^j x_1` is the first ALSSM and :math:`s_j^{(2)} = C_2 A_2^j x_2` the second.

    """
    alssm_type_str = "product"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, alssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        alssms : tuple
            Set of autonomous linear state space models
        G : array_like or None
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label

        Examples
        --------
        Multiply two ALSSMs.

        >>> import lmlib as lm
        >>>
        >>> alssm_p = lm.AlssmPoly(Q=2, label='poly')
        >>> alssm_s = lm.AlssmSin(omega=0.5, rho=0.2, label='sin')
        >>> alssm = lm.AlssmProd((alssm_s, alssm_p), label="multi")
        >>> alssm.update()
        >>> print('<--PRINT-->')
        >>> print(alssm)
        >>> print('<--DUMP-->')
        >>> print(alssm.dump_tree())
        <--PRINT-->
        A =
        [[ 0.17551651  0.17551651  0.17551651 -0.09588511 -0.09588511 -0.09588511]
         [ 0.          0.17551651  0.35103302 -0.         -0.09588511 -0.19177022]
         [ 0.          0.          0.17551651 -0.         -0.         -0.09588511]
         [ 0.09588511  0.09588511  0.09588511  0.17551651  0.17551651  0.17551651]
         [ 0.          0.09588511  0.19177022  0.          0.17551651  0.35103302]
         [ 0.          0.          0.09588511  0.          0.          0.17551651]]
        C =
        [1. 0. 0. 0. 0. 0.]
        <--DUMP-->
        └-product Alssm(multi, A:(6, 6), C:(6,))
          └-sinusoidal Alssm(sin, A:(2, 2), C:(2,))
          └-polynomial Alssm(poly, A:(3, 3), C:(3,))

        """
        N = 0
        L = 0
        for alssm in alssms:
            N *= alssm.N
            L *= alssm.output_count
        super(AlssmProd, self).__init__(
            alssms, G, np.zeros((N, N)), np.zeros((L, N)), label
        )
        self.update()

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        isMO = False
        for alssm in self.alssms:
            alssm.update()
            isMO = isMO or alssm.hasVectorOutput

        self.A = lm.kron_sequence(tuple([alssm.A for alssm in self.alssms]))

        if isMO:
            if self.G is not None:
                self.C = lm.kron_sequence(
                    [
                        g * np.atleast_2d(alssm.C)
                        for g, alssm in zip(self.G, self.alssms)
                    ]
                )
            else:
                self.C = lm.kron_sequence(
                    [np.atleast_2d(alssm.C) for alssm in self.alssms]
                )
        else:
            if self.G is not None:
                self.C = lm.kron_sequence(
                    tuple([g * alssm.C for g, alssm in zip(self.G, self.alssms)])
                )
            else:
                self.C = lm.kron_sequence(tuple([alssm.C for alssm in self.alssms]))
        self._check_dimensions()
        self.isUpdated = True


# Linear state space models


class Lssm(_ModelBase):
    r"""
    Linear state space model (LSSM) class

    This class holds the parameters of a single- or multi-output discrete-time linear state space model (LSSM).
    The output of such an LSSM is defined recursively as

    .. math::
       x[k+1] &= Ax[k] + Bu[k]

       y[k] &= Cx[k] + Du[k] ,

    with fixed parameters :math:`A \in \mathbb{R}^{N\times N}, B \in \mathbb{R}^{N\times M},
    C \in \mathbb{R}^{L\times N}, D \in
    \mathbb{R}^{L\times M}`,
    discrete time index :math:`k`,
    output signal :math:`y`,
    state vector :math:`x`,
    and an input signal :math:`u`.


    | :math:`N` : LSSM order
    | :math:`M` : input dimension
    | :math:`L` : output dimension

    """
    lssm_type_str = "native"
    """str: String representing the current class type (in a well-readable form)"""

    def __init__(self, A, B, C, D, label="label missing"):
        """
        Parameters
        ----------
        A : array_like
            Sets the state transition matrix.
            `A` needs a square shape like `(N, N)`, where `N` is the model order.
        B : array_like
            Sets the input matrix.
            If `B` has shape `(N,)`, the model is **single-input**.
            Two-dimensional input matrix of shape `(N, M)` leads to a **multi-output** model.
            See :ref:`lmlib_single_multi_input_output_lssm`.
        C : array_like
            Sets the output matrix.
            If `C` has shape `(N,)`, the model is **single-output**.
            Two-dimensional output matrix of shape `(L, N)` leads to a **multi-output** model.
            See :ref:`single_multi_output_description`.
        D : array_like, float
            Sets the direct transition (feed-through) matrix.

            - For **single-input** and **single-output** models D is scalar (`shape=()`).
            - For **multi-input** and **single-output** models D takes s shape of `(M,)`.
            - For **single-input** and **multi-output** models D takes s shape of `(L,)`.
            - For **multi-input** and **multi-output** models D takes s shape of `(L, M)`.
        label : str
            Sets the model label.


        ---

        | |def_N|
        | |def_L|
        | |def_M_input|


        .. _single_multi_output_description:

        Notes
        -----
        All LSSM classes equally provide single- and multi-channel systems,
        chosen and set by the structure of :math:`C`.
        As this choice directly defines the structure of most of the output parameters,
        this should be considered carefully. See more in :ref:`lmlib_single_multi_input_output_lssm`.

        - If :math:`C` is a one-dimensional array, then the LSSM object is denoted as a *single-output*
          and results in the scalar output :math:`y`.

        - If :math:`C` is a two-dimensional array, then the LSSM object is denoted as a *multi-output*
          and results in a one-dimensional output :math:`y` vector.
          signals :math:`y`,
          (regardless of whether the effective number of outputs is only 1, i.e. :math:`C` has only one row).


        Examples
        --------
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> B = [3, 0]
        >>> D = [0]
        >>> lssm = lm.Lssm(A, B, C, D)
        >>> lssm.update()
        >>> print(lssm)
        A =
        [[1 1]
         [0 1]]
        B =
        [3 0]
        C =
        [1 0]
        D =
        [0]

        """
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)
        self.D = np.asarray(D)
        self.label = label
        self.update()

    def __repr__(self):
        return (
            "Lssm : "
            + self.lssm_type_str
            + ", A:"
            + str(self.A.shape)
            + ", B:"
            + str(self.B.shape)
            + ", C:"
            + str(self.C.shape)
            + ", D:"
            + str(self.D.shape)
            + ", label: "
            + self.label
        )

    def __str__(self):
        return "A = \n{}\nB = \n{}\nC = \n{}\nD = \n{}".format(
            self.A.__str__(), self.B.__str__(), self.C.__str__(), self.D.__str__()
        )

    def _check_dimensions(self):
        assert self.A.shape[0] == self.A.shape[0], (
            "State Transition matrix is not square."
            + "\n"
            + "Found A shape: "
            + str(self.A.shape)
        )
        assert self.C.shape[-1] == self.A.shape[0], (
            "Last dimension of output matrix doesn't match with system order."
            + "\n"
            + "Found A shape: "
            + str(self.A.shape)
            + "\n"
            + "Found C shape: "
            + str(self.C.shape)
        )
        assert self.B.shape[0] == self.A.shape[0], (
            "First dimension of input matrix doesn't match with system order."
            + "\n"
            + "Found A shape: "
            + str(self.A.shape)
            + "\n"
            + "Found B shape: "
            + str(self.B.shape)
        )
        if self.hasVectorOutput:
            assert (
                self.C.shape[0] == self.D.shape[0]
            ), "First dimension of output matrix doesn't match with first " "dimension of direct transition matrix." + "\n" + "Found C shape: " + str(
                self.C.shape
            ) + "\n" + "Found D shape: " + str(
                self.D.shape
            )
        if self.hasVectorInput:
            assert (
                self.B.shape[-1] == self.D.shape[-1]
            ), "Last dimension of input matrix doesn't match with last " "dimension of direct transition matrix." + "\n" + "Found B shape: " + str(
                self.B.shape
            ) + "\n" + "Found D shape: " + str(
                self.D.shape
            )

    def update(self):
        """
        |fct_doc_lssm_update_header|

        |fct_doc_lssm_update|
        """
        self._check_dimensions()

    def eval(self, xs, us=None):
        r"""
        Evaluate linear state space model

        The evaluation of a LSSM of order :math:`N` needs a sequence of states space vectors
        :math:`x_k \in \mathbb{R}^{N}` and a input signal :math:`u_k \in \mathbb{R}^M`,
        where :math:`M` is the number of LSSM inputs.
        The output :math:`y_k \in \mathbb{R}^L`, with :math:`L` LSSM outputs, is calucalated by

        .. math::

            y_k = Cx_k + Du_k

        where :math:`C \in \mathbb{R}^{L \times N}` is the output matrix
        and :math:`D \in \mathbb{R}^{L \times M}` the direct transition matrix.

        Parameters
        ----------
        xs : array_like
            Sequence of state space vectors of shape `(K, N)`
        us : array_like
            Input signal of shape `(K, M)`, optional

        Returns
        -------
        y : :class:`~numpy.ndarray`
            Output of shape `(K, L)`, if `u` is not given `eval()` returns `y` as :math:`y_k = Cx_k`.


        ---

        | |def_K|
        | |def_N|
        | |def_L|
        | |def_M_input|


        See Also
        --------
        Alssm.eval

        """
        if us is None:
            return np.einsum("...i, hi...->h...", self.C, xs)
        return np.stack([self.C @ x + self.D @ u for x, u in zip(xs, us)], axis=0)

    @property
    def B(self):
        """:class:`~numpy.ndarray` : Input matrix :math:`B \\in \\mathbb{R}^{N \\times N}`"""
        return self._B

    @B.setter
    def B(self, B):
        self._B = B

    @property
    def D(self):
        """:class:`~numpy.ndarray` : Direct transition (feed-through) matrix :math:`D \\in \\mathbb{R}^{L \\times M}`"""
        return self._D

    @D.setter
    def D(self, D):
        if self.hasVectorOutput or self.hasVectorInput:
            assert D.shape is not (), (
                "Direct transition matrix needs to be an array with least one dimension "
                "when the input and output aren't scalar."
            )
        self._D = D

    @property
    def input_count(self):
        """int : Number of inputs :math:`M`"""
        if self.B.ndim is 0:
            return 0
        elif self.B.ndim is 1:
            return 1
        else:
            return self.B.shape[-1]

    @property
    def hasVectorInput(self):
        """bool : Returns True if the model output has at least one dimension."""
        return True if self.B.ndim is 2 else False


class LssmPoly(Lssm):
    """Linear state space model with an impulse response of a polynomial sequence

    Discrete-time linear state space model generating output sequences of shape of a polynomial of order `Q`.
    """

    lssm_type_str = "polynomial"
    """str: String representing the current class type (in a well-readable form)"""

    Q = 0
    """int : Polynomial order :math:`Q`"""

    def __init__(self, Q, B, C, D, label="label missing"):
        """
        Parameters
        ----------
        Q : int
            Sets the polynomial order (polynomial degree -1). The LSSM order `N` is ``Q+1``.
        B : array_like
            Sets the input matrix.
            If `B` has shape `(N,)`, the model is **single-input**.
            Two-dimensional input matrix of shape `(N, M)` leads to a **multi-output** model.
            See :ref:`lmlib_single_multi_input_output_lssm`.
        C : array_like
            Sets the output matrix.
            If `C` has shape `(N,)`, the model is **single-output**.
            Two-dimensional output matrix of shape `(L, N)` leads to a **multi-output** model.
            See :ref:`single_multi_output_description`.
        D : array_like, float
            Sets the direct transition (feed-through) matrix.

            - For **single-input** and **single-output** models D is scalar (`shape=()`).
            - For **multi-input** and **single-output** models D takes s shape of `(M,)`.
            - For **single-input** and **multi-output** models D takes s shape of `(L,)`.
            - For **multi-input** and **multi-output** models D takes s shape of `(L, M)`.
        label : str
            Sets the model label.

        See Also
        --------
        Lssm, AlssmPoly

        Examples
        --------
        Setting up a 3. order polynomial, linear state space model.

        >>> import lmlib as lm
        >>>
        >>> Q = 3
        >>> B = [0, 0, 0, 1]
        >>> C = [1, 0, 0, 0]
        >>> D = 0
        >>> lssm = lm.LssmPoly(Q, B, C, D)
        >>> lssm.update()
        >>> print(lssm)
        A =
        [[1 1 1 1]
         [0 1 2 3]
         [0 0 1 3]
         [0 0 0 1]]
        B =
        [0 0 0 1]
        C =
        [1 0 0 0]
        D =
        0
        """
        assert Q >= 0 and isinstance(
            Q, int
        ), "Polynomial order has to be a positive integer."
        self.Q = Q
        A = np.zeros((Q + 1, Q + 1))
        super(LssmPoly, self).__init__(A, B, C, D, label=label)
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        self.A = pascal(self.Q + 1, kind="upper")
        self._check_dimensions()


class LssmSin(Lssm):
    """Linear state space model with an impulse response of a (damped) sinusoidal sequence.

    Discrete-time linear state space model generating output sequences of sinusoidal shape with decay factor `rho` and
    discrete-time frequency `omega`.
    """

    lssm_type_str = "sinusoidal"
    """str: String representing the current class type (in a well-readable form)"""

    omega = 0.0
    """float : frequency factor :math:`\\omega = 2\\pi f_s`"""
    rho = 0.0
    """float : Decay factor :math:`\\rho`"""

    def __init__(self, omega, rho, B, C, D, label="label missing"):
        """
        Parameters
        ----------
        omega : float
            Sets the frequency :math:`\\omega = 2\\pi f_s`
        rho : float
            Sets the decay factor
        B : array_like
            Sets the input matrix.
            If `B` has shape `(2,)`, the model is **single-input**.
            Two-dimensional input matrix of shape `(2, M)` leads to a **multi-output** model.
            See :ref:`lmlib_single_multi_input_output_lssm`.
        C : array_like
            Sets the output matrix.
            If `C` has shape `(2,)`, the model is **single-output**.
            Two-dimensional output matrix of shape `(L, 2)` leads to a **multi-output** model.
            See :ref:`single_multi_output_description`.
        D : array_like, float
            Sets the direct transition (feed-through) matrix.

            - For **single-input** and **single-output** models D is scalar (`shape=()`).
            - For **multi-input** and **single-output** models D takes s shape of `(M,)`.
            - For **single-input** and **multi-output** models D takes s shape of `(L,)`.
            - For **multi-input** and **multi-output** models D takes s shape of `(L, M)`.
        label : str
            Sets the model label.

        See Also
        --------
        Lssm, AlssmSin

        Examples
        --------
        Setting up a sinusoidal, linear state space model.

        >>> import lmlib as lm
        >>>
        >>> omega=0.1
        >>> rho=0.9
        >>> Q = 3
        >>> B = [1, 1]
        >>> C = [0, 1]
        >>> D = 0.1
        >>> lssm = lm.LssmSin(omega, rho, B, C, D)
        >>> lssm.update()
        >>> print(lssm)
        A =
        [[ 0.89550375 -0.08985007]
         [ 0.08985007  0.89550375]]
        B =
        [1 1]
        C =
        [0 1]
        D =
        0.1
        """
        self.omega = omega
        self.rho = rho
        A = np.zeros((2, 2))
        super(LssmSin, self).__init__(A, B, C, D, label=label)
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        c, s = np.cos(self.omega), np.sin(self.omega)
        self.A = self.rho * np.array([[c, -s], [s, c]])
        self._check_dimensions()


class LssmExp(Lssm):
    """Exponential linear state space model

    Discrete-time linear state space model generating output sequences of exponentially decaying shape with decay
    factor :math:`\\gamma`.
    """

    lssm_type_str = "exponential"
    """str: String representing the current class type (in a well-readable form)"""

    gamma = 0.0
    """float :  decay factor per sample :math:`\\gamma`"""

    def __init__(self, gamma, B, C, D, label="label missing"):
        """
        Parameters
        ----------
        gamma : float
            Sets decay factor per sample ( < 1.0 : left-side decaying; > 1.0 : right-side decaying)
        B : array_like
            Sets the input matrix.
            If `B` has shape `(1,)`, the model is **single-input**.
            Two-dimensional input matrix of shape `(1, M)` leads to a **multi-output** model.
            See :ref:`lmlib_single_multi_input_output_lssm`.
        C : array_like
            Sets the output matrix.
            If `C` has shape `(1,)`, the model is **single-output**.
            Two-dimensional output matrix of shape `(L, 1)` leads to a **multi-output** model.
            See :ref:`single_multi_output_description`.
        D : array_like, float
            Sets the direct transition (feed-through) matrix.
        label : str
            Sets the label of the LSSM.

        See Also
        --------
        Lssm, AlssmExp

        Examples
        --------
        Setting up an exponential, linear state space model.

        >>> import lmlib as lm
        >>>
        >>> gamma=0.8
        >>> B = [1]
        >>> C = [0.9]
        >>> D = 0.1
        >>> lssm = lm.LssmExp(gamma, B, C, D)
        >>> lssm.update()
        >>> print(lssm)
        A =
        [[0.8]]
        B =
        [1]
        C =
        [0.9]
        D =
        0.1

        """
        self.gamma = gamma
        A = np.zeros((1, 1))
        super(LssmExp, self).__init__(A, B, C, D, label=label)
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        self.A = np.array([[self.gamma]])
        self._check_dimensions()


# LSSM modifiers


class LssmContainer(Lssm, _ModelContainer, ABC):
    """Abstract class for LSSM Container"""

    def __init__(self, lssms, G, A, B, C, D, label):
        """
        Parameters
        ----------
        lssms : tuple
            Set of autonomous linear state space models
        A : array_like
            Sets the state transition matrix.
            `A` needs a square shape like `(N, N)`, where `N` is the model order.
        B : array_like
            Sets the input matrix.
            If `B` has shape `(N,)`, the model is **single-input**.
            Two-dimensional input matrix of shape `(N, M)` leads to a **multi-output** model.
            See :ref:`lmlib_single_multi_input_output_lssm`.
        C : array_like
            Sets the output matrix.
            If `C` has shape `(N,)`, the model is **single-output**.
            Two-dimensional output matrix of shape `(L, N)` leads to a **multi-output** model.
            See :ref:`single_multi_output_description`.
        D : array_like, float
            Sets the direct transition (feed-through) matrix.

            - For **single-input** and **single-output** models D is scalar (`shape=()`).
            - For **multi-input** and **single-output** models D takes s shape of `(M,)`.
            - For **single-input** and **multi-output** models D takes s shape of `(L,)`.
            - For **multi-input** and **multi-output** models D takes s shape of `(L, M)`.
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        self.lssms = lssms
        self.G = G
        super(LssmContainer, self).__init__(A, B, C, D, label=label)

    @property
    def lssms(self):
        """tuple : Set of linear state space models"""
        return self._models

    @lssms.setter
    def lssms(self, lssms):
        for i, lssm in enumerate(lssms):
            assert isinstance(lssm, (Lssm, Alssm)), (
                "The object at position "
                + str(i)
                + " in lssms is not of type Lssm or Alssm."
                + "\n"
                + "Found element of type: "
                + lssm.__name__
            )
            self._models = lssms


class LssmStacked(LssmContainer):
    """
    Stacking linear state space models to one single LSSM

    The outputs staying individual.
    """

    lssm_type_str = "stacked"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, lssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        lssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        N = 0
        L = 0
        M = 0
        for lssm in lssms:
            N += lssm.N
            L += lssm.output_count
            M += lssm.input_count if isinstance(lssm, Lssm) else 0
        super(LssmStacked, self).__init__(
            lssms,
            G,
            np.zeros((N, N)),
            np.zeros((N, L)),
            np.zeros((L, N)),
            np.zeros((L, M)),
            label,
        )
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        isMI = False
        isMO = False
        B_list = []
        D_list = []
        for lssm in self.lssms:
            lssm.update()
            isMI = isMI or lssm.hasVectorInput if isinstance(lssm, Lssm) else isMI
            isMO = isMO or lssm.hasVectorOutput

            if isinstance(lssm, Lssm):
                B_list.append(
                    lssm.B if lssm.hasVectorInput else np.atleast_2d(lssm.B).T
                )
                if lssm.hasVectorInput and not lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D))
                elif not lssm.hasVectorInput and lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D).T)
                else:
                    D_list.append(lssm.D)
            else:
                B_list.append(np.zeros((lssm.N, 0)))
                D_list.append(np.zeros((lssm.output_count, 0)))

        self.A = block_diag(*[lssm.A for lssm in self.lssms])
        self.B = (
            block_diag(*B_list) if isMI else np.squeeze(block_diag(*B_list), axis=-1)
        )
        self.D = (
            block_diag(*D_list) if isMI else np.squeeze(block_diag(*D_list), axis=-1)
        )

        if self.G is not None:
            self.C = block_diag(*[g * lssm.C for g, lssm in zip(self.G, self.lssms)])
        else:
            self.C = block_diag(*[lssm.C for lssm in self.lssms])
        self._check_dimensions()


class LssmStackedSO(LssmContainer):
    """
    Stacking linear state space models to one single LSSM with summed outputs
    """

    lssm_type_str = "stacked"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, lssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        lssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        L = lssms[0].output_count
        N = 0
        M = 0
        for i, lssm in enumerate(lssms):
            assert lssm.output_count == L, (
                "The output dimension of the lssm at position "
                + str(i)
                + " in lssms doesn't match with the first."
                + "\n"
                + "Expected output dimension: "
                + str(L)
                + "\n"
                + "Found output dimension of lssm at position "
                + str(i)
                + ": "
                + str(lssm.output_count)
            )
            N += lssm.N
            M += lssm.input_count if isinstance(lssm, Lssm) else 0
        super(LssmStackedSO, self).__init__(
            lssms,
            G,
            np.zeros((N, N)),
            np.zeros((N, L)),
            np.zeros((L, N)),
            np.zeros((L, M)),
            label,
        )
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        isMI = False
        isMO = False
        B_list = []
        D_list = []
        for lssm in self.lssms:
            lssm.update()
            # check if stacked lssm is gonna be multi-in/output
            isMI = isMI or lssm.hasVectorInput if isinstance(lssm, Lssm) else isMI
            isMO = isMO or lssm.hasVectorOutput

            # creating list for bock diagonals to match LSSM (B and D) with ALSSM
            if isinstance(lssm, Lssm):
                B_list.append(
                    lssm.B if lssm.hasVectorInput else np.atleast_2d(lssm.B).T
                )
                if lssm.hasVectorInput and not lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D))
                elif not lssm.hasVectorInput and lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D).T)
                else:
                    D_list.append(lssm.D)
            else:
                B_list.append(np.zeros((lssm.N, 0)))

        # creating block diagonals and squeeze if not multi input or output
        self.A = block_diag(*[lssm.A for lssm in self.lssms])
        self.B = (
            block_diag(*B_list) if isMI else np.squeeze(block_diag(*B_list), axis=-1)
        )
        self.D = (
            block_diag(*D_list) if isMI else np.squeeze(block_diag(*D_list), axis=-1)
        )
        if not isMO:
            self.D = np.squeeze(self.D, axis=0)

        # create output matrix with scale factors
        if isMO:
            if self.G is not None:
                self.C = np.concatenate(
                    [g * np.atleast_2d(lssm.C) for g, lssm in zip(self.G, self.lssms)],
                    axis=-1,
                )
            else:
                self.C = np.concatenate(
                    [np.atleast_2d(lssm.C) for lssm in self.lssms], axis=-1
                )
        else:
            if self.G is not None:
                self.C = np.concatenate(
                    [g * lssm.C for g, lssm in zip(self.G, self.lssms)], axis=-1
                )
            else:
                self.C = np.concatenate([lssm.C for lssm in self.lssms], axis=-1)
        self._check_dimensions()


class LssmStackedCI(LssmContainer):
    """
    Stacking linear state space models to one single LSSM with common inputs

    The outputs staying individual.
    """

    lssm_type_str = "stacked-ci"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, lssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        lssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        L = 0
        N = 0
        M = lssms[0].input_count
        for i, lssm in enumerate(lssms):
            if isinstance(lssm, Lssm):
                assert (
                    lssm.input_count == M
                ), "The input dimension of the lssm at position " + str(
                    i
                ) + " in lssms doesn't match " "with the dimension of the first lssm." + "\n" + "Expected input dimension: " + str(
                    L
                ) + "\n" + "Found input dimension of lssm at position " + str(
                    i
                ) + ": " + str(
                    lssm.input_count
                )
            N += lssm.N
            L += lssm.output_count if isinstance(lssm, Lssm) else 0
        super(LssmStackedCI, self).__init__(
            lssms,
            G,
            np.zeros((N, N)),
            np.zeros((N, L)),
            np.zeros((L, N)),
            np.zeros((L, M)),
            label,
        )
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        isMI = False
        isMO = False
        B_list = []
        D_list = []
        M = self.input_count
        for lssm in self.lssms:
            lssm.update()
            # check if stacked lssm is gonna be multi-in/output
            isMI = isMI or lssm.hasVectorInput if isinstance(lssm, Lssm) else isMI
            isMO = isMO or lssm.hasVectorOutput

            # creating list for bock diagonals to match LSSM (B and D) with ALSSM
            if isinstance(lssm, Lssm):
                B_list.append(
                    lssm.B if lssm.hasVectorInput else np.atleast_2d(lssm.B).T
                )
                if lssm.hasVectorInput and not lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D))
                elif not lssm.hasVectorInput and lssm.hasVectorOutput:
                    D_list.append(np.atleast_2d(lssm.D).T)
                else:
                    D_list.append(lssm.D)
            else:
                B_list.append(np.zeros((lssm.N, M)))
                D_list.append(np.zeros((lssm.output_count, M)))

        # creating block diagonals and squeeze if not multi input or output
        self.A = block_diag(*[lssm.A for lssm in self.lssms])
        self.B = np.vstack(B_list) if isMI else np.squeeze(np.vstack(B_list), axis=-1)
        self.D = np.vstack(D_list) if isMI else np.squeeze(np.vstack(D_list), axis=-1)
        if not isMO:
            self.D = np.squeeze(self.D, axis=0)

        # create output matrix with scale factors
        if self.G is not None:
            self.C = block_diag(*[g * lssm.C for g, lssm in zip(self.G, self.lssms)])
        else:
            self.C = block_diag(*[lssm.C for lssm in self.lssms])
        self._check_dimensions()


class LssmStackedCISO(LssmContainer):
    """
    Stacking (A)LSSMs to one single LSSM with common inputs and summed outputs
    """

    lssm_type_str = "stacked-ci"
    """str: string representing the current class type (in a well-readable form)"""

    def __init__(self, lssms, G=None, label="label missing"):
        """
        Parameters
        ----------
        lssms : tuple
            Set of autonomous linear state space models
        G : list, tuple
            Set of scalars. Each scales the output matrix of the ALSSM in `alssms`.
        label : str
            Sets the model label
        """
        L = lssms[0].output_count
        N = 0
        M = lssms[0].input_count
        for i, lssm in enumerate(lssms):
            if isinstance(lssm, Lssm):
                assert (
                    lssm.input_count == M
                ), "The input dimension of the lssm at position " + str(
                    i
                ) + " in lssms doesn't match " "with the dimension of the first lssm." + "\n" + "Expected input dimension: " + str(
                    L
                ) + "\n" + "Found input dimension of lssm at position " + str(
                    i
                ) + ": " + str(
                    lssm.input_count
                )
            assert lssm.output_count == L, (
                "The output dimension of the lssm at position "
                + str(i)
                + " in lssms doesn't match with the first."
                + "\n"
                + "Expected output dimension: "
                + str(L)
                + "\n"
                + "Found output dimension of lssm at position "
                + str(i)
                + ": "
                + str(lssm.output_count)
            )
            N += lssm.N
        super(LssmStackedCISO, self).__init__(
            lssms,
            G,
            np.zeros((N, N)),
            np.zeros((N, L)),
            np.zeros((L, N)),
            np.zeros((L, M)),
            label,
        )
        self.update()

    def update(self):
        """Initializes state transition matrix and checks state space parameter validity."""
        isMI = False
        isMO = False
        B_list = []
        M = self.input_count
        for lssm in self.lssms:
            lssm.update()
            # check if stacked lssm is gonna be multi-in/output
            isMI = isMI or lssm.hasVectorInput if isinstance(lssm, Lssm) else isMI
            isMO = isMO or lssm.hasVectorOutput

            # creating list for bock diagonals to match LSSM (B and D) with ALSSM
            if isinstance(lssm, Lssm):
                B_list.append(
                    lssm.B if lssm.hasVectorInput else np.atleast_2d(lssm.B).T
                )
                if lssm.hasVectorInput and not lssm.hasVectorOutput:
                    self.D += np.atleast_2d(lssm.D)
                elif not lssm.hasVectorInput and lssm.hasVectorOutput:
                    self.D += np.atleast_2d(lssm.D).T
                else:
                    self.D += lssm.D
            else:
                B_list.append(np.zeros((lssm.N, M)))

        # creating block diagonals and squeeze if not multi input or output
        self.A = block_diag(*[lssm.A for lssm in self.lssms])
        self.B = np.vstack(B_list) if isMI else np.squeeze(np.vstack(B_list), axis=-1)
        if not isMI:
            self.D = np.squeeze(self.D, axis=-1)
        if not isMO:
            self.D = np.squeeze(self.D, axis=0)

        # create output matrix with scale factors
        if isMO:
            if self.G is not None:
                self.C = np.concatenate(
                    [g * np.atleast_2d(lssm.C) for g, lssm in zip(self.G, self.lssms)],
                    axis=-1,
                )
            else:
                self.C = np.concatenate(
                    [np.atleast_2d(lssm.C) for lssm in self.lssms], axis=-1
                )
        else:
            if self.G is not None:
                self.C = np.concatenate(
                    [g * lssm.C for g, lssm in zip(self.G, self.lssms)], axis=-1
                )
            else:
                self.C = np.concatenate([lssm.C for lssm in self.lssms], axis=-1)
        self._check_dimensions()
