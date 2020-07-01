# -*- coding: utf-8 -*-
# Author: Waldmann Frédéric, Wildhaber Reto
r"""
Squared Error Cost Functions

Signal models composed based on ALSSMs, composed of one or  multiple cost segments.

Signal models composed of one or  multiple cost segments (i.e., weighted time intervals) and Autonomous Linear State
Space Models (ALSSMs)


This submodule provides:
------------------------

* :class:`CostSeg` :  Description of a signal model, composed of a ALSSM and a segment (window). * :class:`CCost` :
Description of signal model, composed of one or multiple ALSSMs, mapped to one or multiple segment (window).

The output sequence of an Autonomous Linear State Space Models (ALSSMs) is fully determined by its initial state
:math:`x` as it has no input. Therefore, ALSSMs are convenient as local signal models while providing recursive
computation rules for the computation of squared signal errors. For example, the squared error between the output of
the ALSSM :math:`cA^ix` and the observations :math:`y_i` over an interval from :math:`a` to :math:`b` is

.. math::

   J(x) = \sum_{i=a}^{b} \big(cA^ix - y_i \big)^2

where :math:`A` and :math:`c` are the ALSSM parameters and :math:`x` its initial state (to be estimated in a least
squares problem), see [Wildhaber2019]_ [Section 4.2] and [Chapter 9].


Constants
---------
.. autosummary::
   :toctree:  _costfunc/_constants

   FORWARD
   BACKWARD

Classes
-------
.. autosummary::
   :toctree:  _costfunc/_constants

   Segment
   CostSegment
   CCost
   SEParam
   SEParamSteadyState

"""

from __future__ import division, absolute_import, print_function

__all__ = [
    "Segment",
    "BACKWARD",
    "FORWARD",
    "get_steady_state_W",
    "CostSegment",
    "CCost",
    "SEParam",
    "SEParamSteadyState",
]

import numpy as np
import lmlib as lm
from numpy.linalg import inv

# Constants

FORWARD = "forward recursion"
"""str : Sets the recursion direction in a segment to forward."""
BACKWARD = "backward recursion"
"""str : Sets the recursion direction in a segment to backward."""
LANGUAGE_ARRAY_OFFSET = 0
"""int : array begin index Python = 0, Matlab = 1"""
MATH_START_INDEX = 1
"""int : math starting index"""
MATH_TO_LANG_OFFSET = -MATH_START_INDEX + LANGUAGE_ARRAY_OFFSET
"""int : Math (starts with indexing 1) to programming arrays at `LANGUAGE_ARRAY_OFFSET`"""


def _W_closed_form(A, C, gamma, a, b):
    N = np.shape(A)[0]
    ATA = np.kron(np.transpose(A), A)
    ATA_a = (
        np.linalg.matrix_power(gamma * np.kron(A.T, A), a)
        if ~(np.isinf(a))
        else np.zeros(N * N)
    )
    ATA_b = (
        np.linalg.matrix_power(gamma * np.kron(A.T, A), b + 1)
        if ~(np.isinf(b))
        else np.zeros(N * N)
    )
    return (
        np.kron(np.eye(N), np.atleast_2d(C))
        @ (inv(np.eye(N * N) - np.dot(gamma, ATA)) @ (ATA_a - ATA_b))
        @ np.kron(np.transpose(np.atleast_2d(C)), np.eye(N))
    )


def get_steady_state_W(cost_model, F=None):
    if isinstance(cost_model, lm.CostSegment):
        return _W_closed_form(
            cost_model.alssm.A,
            cost_model.alssm.C,
            cost_model.segment.gamma,
            cost_model.segment.a,
            cost_model.segment.b,
        )

    elif isinstance(cost_model, lm.CCost):
        W = []
        for segment, f in zip(cost_model.segments, np.transpose(F)):
            alssm = lm.AlssmStackedSO(cost_model.alssms, G=f)
            W.append(
                _W_closed_form(alssm.A, alssm.C, segment.gamma, segment.a, segment.b)
            )
        return np.sum(W, axis=0)
    else:
        raise TypeError("Cost Model is not of type CostSegment or CCost")


# Segment


class Segment:
    r"""Segment with a weighting window of finite or infinite interval borders.

    Segments are commonly used in conjunction with one or multiple ALSSMs,
    which together define a model as used for cost functions in ``CostSeg`` or ``CCost``.
    The window of a segment is either of exponentially decaying shape or, more general,
    defined by the output of its own ALSSM, a so-called `window ALSSM`.
    A segment also determines, based on the chosen window shape,
    the required computation to gain stable recursions in any subsequent recursive cost computations.

    Used as a squared error cost function, a cost segment writes as the cost function:

    .. math::
        J_k(x) = \sum_{i=k+a}^{k+b} \gamma^{i-k-\delta}\big(CA^{i-k}x - y_i\big)^2 \ ,

    and when using sample weights :math:`v_k` as the cost function, according [Wildhaber2018]_ [Equation (14)],

    .. math::
        J_k(x) = \sum_{i=k+a}^{k+b} v_k  {\alpha}_{k+\delta}(k+\delta) \big(CA^{i-k}x - y_i\big)^2 \ ,

    with
    with the sample weights :math:`v_k` (not yet implemented <TODO>)
    and the window weight :math:`\alpha_k(j)` which depends on the sample weights.

    See also [Wildhaber2018]_ [Wildhaber2019]_

    """

    def __init__(self, a, b, direction, g, delta=0, label="label missing"):
        r"""
        Parameters
        ----------
        a : int
            left boundary of the segment's interval
        b : int
            right boundary of the segment's interval
        g : unsigned int
            :math:`g > 0`,
            Sets effective sample number under the window.
            The effective sample number corresponds to the (weighted) number of samples under
            the exponentially decaying window
            (to the left or right of :math:`k+ \delta`, depending on the computation direction)
        direction : str
            Defines the segment's recursion computation `direction`

            - ``lm.FORWARD`` use forward computation with forward recursions
            - ``lm.BACKWARD`` use backward computation with backward recursions
        delta : int
            Shifts the window such that the window at relative index :math:`\delta`  (absolute index :math:`k+\delta`)
            is `1`, i.e.,
            :math:`{\alpha}_{k+\delta}(k+\delta)=1`
        label : str
            Segment label

        Notes
        -----
        Both boundaries `a` and `b` are included in the segment, i.e., the sum runs over :math:`k \in [a,b] ` and
        includes `b-a+1` samples.


        """
        self.a = a
        self.b = b
        self.direction = direction
        self.g = g
        self.delta = delta
        self.label = label

        # window function: constant decay which is independent of sample weights w_k
        if self.direction == FORWARD:
            self.gamma = self.g / (self.g - 1)
            assert not np.isinf(
                self.b
            ), "Right boundary is infinite what is not allowed with forward recursions!"
        elif self.direction == BACKWARD:
            self.gamma = (self.g / (self.g - 1)) ** (-1)
            assert not np.isinf(
                self.a
            ), "Left boundary is infinite what is not allowed with backward recursions!"
        else:
            raise ValueError("Unknown direction parameter: " + str(self.direction))

    def __repr__(self):
        return "a = {}\nb = {}\ng = {}\ndirection = {}\ndelta = {},\ngamma = {}".format(
            self.a, self.b, self.g, self.direction, self.delta, self.gamma
        )

    def __str__(self):
        return "Segment : a:{}, b:{}, {}, g:{}, label: {}".format(
            self.a, self.b, self.direction, self.g, self.label
        )

    @property
    def a(self):
        """int : right boundary of the segment's interval :math:`a`"""
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def b(self):
        """int : left boundary of the segment's interval :math:`b`"""
        return self._b

    @b.setter
    def b(self, b):
        self._b = b

    @property
    def g(self):
        """float : Effective number of samples :math:`g`, setting the window with

        The effective number of samples :math:`g` is used to derive
        and set the window decay factor :math:`\\gamma` internally.

        See Also
        --------
        :attr:`Segment.gamma` and [Wildhaber2018] [Section III.A]

        """
        return self._g

    @g.setter
    def g(self, g):
        self._g = g

    @property
    def direction(self):
        """str : Sets the segment's recursion computation `direction`

            - ``lm.FORWARD`` use forward computation with forward recursions
            - ``lm.BACKWARD`` use backward computation with backward recursions
        """
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = direction

    @property
    def delta(self):
        """int : Relative window shift :math:`\\delta`"""
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._delta = delta

    @property
    def label(self):
        """str : Label of the segment"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def gamma(self):
        r"""float : Window decay factor :math:`\gamma`

        Window decay factor :math:`\gamma` is set internally on the initialization of a new segment object
        and is derived from the *effective number of samples* ``Segment.g`` as follows:

        - for a segment with forward recursions: :math:`\gamma = \frac{g}{g-1}`
        - for a segment with forward recursions: :math:`\gamma = \big(\frac{g}{g-1}\big)^{-1}`

        See Also
        --------
        [Wildhaber2018] [Table IV]

        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    def window(self, ref=(), merge_k="none"):
        """Returns window weights of the segment

        Parameters
        ----------
        ref : tuple
            TODO
        merge_k : str
            'none', 'max-win', 'sum'

        Returns
        -------
        wins : :class:~`numpy.ndarray`
            shape=(`K`, [`len(ks)`]),

        """
        if len(ref) is 0:
            if np.isinf(self.a) or np.isinf(self.b):
                raise ValueError(
                    "Infinite boundaries just available with ref argument."
                )
            win = [
                self._window_alpha_func(self.delta, i)
                for i in np.arange(self.a, self.b + 1)
            ]
            if merge_k != "none":
                return np.asarray(win)
            return np.asarray([win])

        ks = ref[0]
        K = ref[1]
        wins = np.zeros((K, len(ks)))

        for n, k in enumerate(ks):
            a_to_b1 = np.arange(max(self.a, -k), min(self.b + 1, K - k))
            wins[k + a_to_b1, n] = [
                self._window_alpha_func(self.delta, i) for i in a_to_b1
            ]
        if merge_k == "max-win":
            return np.amax(wins, axis=1)
        elif merge_k == "sum":
            return np.sum(wins, axis=1)
        elif merge_k == "none":
            return np.asarray(wins)
        else:
            raise ValueError("unknown merge_k value")

    def _window_alpha_func(self, k, i):
        """window weight function as in [Wildhaber2019]_"""
        if i < k:
            return np.power(1 / self.gamma, k - i)
        elif i > k:
            return np.power(self.gamma, i - k)
        else:  # i == k
            return 1


class SegmentDynWin:
    pass


# Cost Models


class CostSegment:
    r"""
    Assignment of a ALSSM to a Segment to define the quadratic costs.

    As seen in [Wildhaber2019]_ [Chapter 9, The Cost Segment] and [Section 4.2.6],
    a cost segment is a quadratic cost function

    .. math::
        J_a^b(k,x,\theta) = \sum_{i=k+a}^{k+b} \alpha_{k+\delta}(i)v_i(y_i - cA^{i-k}x)^2

    over a fixed interval :math:`\{a, \dots, b\}` with :math:`a \in \mathbb{Z} \cup \{ - \infty \}`,
    :math:`b \in \mathbb{Z} \cup \{ + \infty\}`, and :math:`a \le b`,
    and with initial state vector :math:`x \in \mathbb{R}^{N \times 1}`.
    """

    def __init__(self, alssm, segment, label="label missing"):
        """
        Parameters
        ----------
        alssm : :class:`~lmlib.statespace.model.Alssm`
            Sets the ALSSM
        segment : :class:`Segment`
            Sets the Segment
        label : str
            Cost segment label

        See Also
        --------
        CCost

        Examples
        --------
        Setup a cost segment with finite boundaries and a line ALSSM.

        >>> import lmlib as lm
        >>>
        >>> A = [[1, 1], [0, 1]]
        >>> C = [1, 0]
        >>> alssm = lm.Alssm(A, C, label="line")
        >>>
        >>> seg = lm.Segment(a=-30, b=0, direction=lm.FORWARD, g=20, label="finite_left")
        >>>
        >>> cost_seg = lm.CostSegment(alssm, seg, label="left_line")
        >>> print(cost_seg)

        """

        assert isinstance(alssm, lm.Alssm), (
            "alssm have to be of type Alssm. \n Found alssm: " + alssm.__name__
        )
        assert isinstance(segment, lm.Segment), (
            "segment have to be of type Seg. \n Found segment : " + segment.__repr__()
        )

        self.alssm = alssm
        self.segment = segment
        self.label = label

    def __str__(self):
        return "CostSegment : label: {} \n  └- {}, \n  └- {} ".format(
            self.label, self.alssm.__repr__(), self.segment
        )

    @property
    def alssm(self):
        """:class:`~lmlib.statespace.model.Alssm` : Autonomous linear state space model"""
        return self._alssm

    @alssm.setter
    def alssm(self, alssm):
        self._alssm = alssm

    @property
    def segment(self):
        """:class:`Segment` : Segment"""
        return self._segment

    @segment.setter
    def segment(self, segment):
        self._segment = segment

    @property
    def label(self):
        """str : Cost segment label"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def window(self, ref=(), merge_k="none"):
        """Returns the segment window

        See Also
        --------
        :meth:`~lmlib.costfunc.Segment.window`
        """
        return self.segment.window(ref, merge_k)

    def trajectory(self, xs, ref=(), thd=0, off_val=np.nan, merge_k="none"):
        r"""Returns the trajectories over `K` samples staring at indices `ks`

        Parameters
        ----------
        xs : array_like
            shape=(`len(ks)`, `N`, [`S`]),
            Array of initial state space vectors
        ref : tuple
            (ks, K) indices, number of samples
        thd : float
            Sets the window-weight threshold. A trajectory sample gets evaluated when its window weight is above the
            threshold, else the sample remains the value `np.nan`
        off_val : scalar
            Values underneath the window weight threshold.
        merge_k : str
            Merges dimensions to a reduced the shape of the output.
            If ``'none'`` the ouput dimension remains. (default)
            If ``'sum'`` the entries in the `ks`-dimension are summed up. (nan values are assumed to be 0)
            If ``'max-win'`` the entry with the higest window weight is chosen.

        Returns
        -------
        out : :class:`~numpy.ndarray`
            shape=(`K`, `len(ks)` [,`L`] [,`S`])
            Returns the trajectories for each state space vector in the list `x_k` with starting positions in `k`.

        Examples
        --------
        See :ref:`lmlib_example_trajectory_on_costsegment`
        See :ref:`lmlib_example_trajectory_on_compositecost`

        ---

        | |def_K|
        | |def_N|
        | |def_L|
        | |def_S|
        """

        if len(ref) is 0:
            if np.isinf(self.segment.a) or np.isinf(self.segment.b):
                raise ValueError(
                    "Creating trajectories with infinite boundaries is just possible with model to output "
                    "reference ks, K."
                )
            a_to_b1 = np.arange(self.segment.a, self.segment.b + 1)
            return self.alssm.eval_at(xs, a_to_b1)

        ks = ref[0]
        K = ref[1]

        trajs = np.vstack(
            [self.alssm.eval_at([x], np.arange(K) - k) for x, k in zip(xs, ks)]
        ).T
        if thd != 0:
            wins = self.segment.window(ref=(ks, K))
            trajs[wins <= thd] = off_val

        if merge_k == "none":
            return trajs
        elif merge_k == "max-win":
            wins = self.segment.window(ref=(ks, K))
            indices_max_win = np.argmax(wins, axis=1)
            return np.choose(indices_max_win, trajs.T).T
        else:
            raise ValueError("unknown merge_k value")

    def eval(self, xs, ref):
        """Evaluates .

        See Also
        --------
        :meth:`lmlib.statespace.model.Alssm.eval`

        """
        return self.alssm.eval(xs, ref)

    def eval_at(self, xs, js, ref):
        """Evaluates the ALSSM for each state space vector in x_k over `K` samples for each index in `ks`.

        See Also
        --------
        :meth:`lmlib.statespace.model.Alssm.eval_at`

        """
        return self.alssm.eval_at(xs, js, ref)


class CCost:
    r"""
    Assignment and mapping of ALSSMs to Segments to define the composite quadratic costs.

    As seen in [Wildhaber2019]_ [Chapter 9, The Composite Cost], composite costs is an interconnection of several
    cost segments. It allows to map different models to multiple segments or vices versa.
    The composite cost is state as

    .. math::
       J(k, x, \Theta) = \sum_{p = 1}^{P}\beta_p J_{a_p}^{b_p}(k, x, \theta_p) \ ,

    where, :math:`\Theta = (\theta_1, \theta_2,\dots, \theta_P)` and  the *segment scalars*
    :math:`\beta_p \in \mathbb{R}_+`.

    """

    def __init__(self, alssms, segments, F, label="label missing"):
        """
        Parameters
        ----------
        alssms : tuple of :class:`~lmlib.statespace.model.Alssm`
            shape=(`M`),
            Set of ALSSMs
        segments : tuple of :class:`Segment`
            shape=(`P`),
            Set of segments
        F : array_like
            shape=(`M`, `P`),
            Mapping / Scaling matrix `F` includes the *segment scalar* and mapping of the models to the segments.
        label : str
            Composite cost label


        ---

        | |def_P|
        | |def_M_models|

        See Also
        --------
        CostSegment

        Examples
        --------
        See here: :ref:`lmlib_composite_cost_example`

        """
        self.alssms = alssms
        self.segments = segments
        self.F = np.asarray(F)
        self.label = label

    @property
    def alssms(self):
        """tuple of :class:`~lmlib.statespace.model.Alssm` : Autonomous linear state space models"""
        return self._alssms

    @alssms.setter
    def alssms(self, alssms):
        self._alssms = alssms

    @property
    def segments(self):
        """tuple of :class:`Segment` : Segments"""
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    @property
    def label(self):
        """str : Composite cost label"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def window(self, ps, ref, merge_k="none", merge_seg="none"):
        wins = np.stack(
            [
                segment.window(ref, merge_k)
                for segment, p in zip(self.segments, ps)
                if p == 1
            ],
            axis=-1,
        )
        if merge_seg == "none":
            return wins
        elif merge_seg == "sum":
            return np.sum(wins, axis=-1)
        elif merge_seg == "max-win":
            return np.amax(wins, axis=-1)
        else:
            raise ValueError("unknown merge_seg value")

    def eval(self, xs, f, ref=()):
        alssm = lm.AlssmStackedSO(self.alssms, G=f)
        alssm.update()
        return alssm.eval(xs, ref)

    def trajectory(
        self, xs, F, ref, thd=0, off_val=np.nan, merge_seg="none", merge_k="none"
    ):
        trajs_seg = []
        for nP, f in enumerate(np.transpose(F)):
            alssm = lm.AlssmStackedSO(self.alssms, G=f)
            alssm.update()
            cost = lm.CostSegment(alssm, self.segments[nP])
            trajs_seg.append(cost.trajectory(xs, ref, thd, off_val, merge_k))
        wins = self.window([1] * len(np.transpose(F)), ref, merge_k)
        if merge_seg == "none":
            trajs = np.stack(trajs_seg, axis=-1)
            trajs[wins < thd] = off_val
            return trajs
        elif merge_seg == "max-win":
            indices_max_win = np.argmax(wins, axis=-1)
            trajs = np.choose(indices_max_win, trajs_seg)
            trajs[np.amax(wins, axis=-1) < thd] = off_val
            return trajs
        else:
            raise ValueError("unknown merge_seg value")


class SEParam:
    """
    Class for computation and storage of intermediate variables, needed to efficiently solve least squares problems
    between a :class:`CCost` model and given observations.

    Main intermediate variables are the covariance `W`, weighed mean `\\xi`, signal energy `\\kappa`, weighted number
    of samples `\\nu`.

    """

    def __init__(self, label="label missing"):
        """

        Parameters
        ----------
        label : str
            FParam label
        """
        self.label = label
        self.W = None
        self.xi = None
        self.kappa = None
        self.v = None

    @property
    def label(self):
        """str : FParam label"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def W(self):
        """:class:`~numpy.ndarray` : Filter Parameter :math:`W`"""
        return self._W

    @W.setter
    def W(self, W):
        self._W = W

    @property
    def xi(self):
        """:class:`~numpy.ndarray` :  Filter Parameter :math:`\\xi`"""
        return self._xi

    @xi.setter
    def xi(self, xi):
        self._xi = xi

    @property
    def kappa(self):
        """:class:`~numpy.ndarray`  : Filter Parameter :math:`\\kappa`"""
        return self._kappa

    @kappa.setter
    def kappa(self, kappa):
        self._kappa = kappa

    @property
    def v(self):
        """:class:`~numpy.ndarray`  : Filter Parameter :math:`v`"""
        return self._v

    @v.setter
    def v(self, v):
        self._v = v

    def allocate(self, K, N, S=None):
        """
        Filter Parameter memory allocation

        Parameters
        ----------
        K : int
            Number of samples
        N : int
            Order of ALSSM
        S : int
            Number of sets


        ---

        | |def_K|
        | |def_N|
        | |def_S|
        """

        self.W = np.zeros((K, N, N))
        self.xi = np.zeros((K, N)) if S is None else np.zeros((K, N, S))
        self.kappa = np.zeros((K,)) if S is None else np.zeros((K, S, S))

    def filter(self, cost_model, y, betas=None):
        """
        Computes the intermediate parameters.

        Computes the intermediate parameters using efficient forward- and backward recursions.
        The results are stored internally in this :class:`FParam` object,
        ready to solve the least squares problem using e.g., :meth:`minimize` or :meth:`minimize_lin`.

        Parameters
        ----------
        cost_model : :class:`CostSegment`, :class:`CCost`
            Cost model
        y : array_like
            shape=(`K`, [`L` [, `S`]]),
            single- or multi-channel input signal
        betas : list, sequence, tuple, None
            shape=(`P`),
            Factor weighting each of the `P` single cost segments in a `CCost`.


        --

        | |def_K|
        | |def_L|
        | |def_S|
        | |def_P|


        Note
        ----
        |note_MC|

        """
        y = np.asarray(y)

        if isinstance(cost_model, lm.CCost):

            if self._W is None or self._xi is None or self._kappa is None:
                N = sum(alssm.N for alssm in cost_model.alssms)
                if y.ndim is 3:
                    self.allocate(y.shape[0], N, y.shape[2])
                else:
                    self.allocate(y.shape[0], N)

            # cost_model is Composite Cost
            if betas is None:
                betas = [1] * len(cost_model.segments)

            for f, segment, beta in zip(cost_model.F.T, cost_model.segments, betas):
                alssm = lm.AlssmStackedSO(cost_model.alssms, G=f)
                alssm.update()
                if segment.direction is lm.BACKWARD:
                    self._backward_recursion(alssm, segment, y, beta)
                elif segment.direction is lm.FORWARD:
                    self._forward_recursion(alssm, segment, y, beta)
                else:
                    ValueError("Segment of cost_model has unknown direction.")
        else:
            if self._W is None or self._xi is None or self._kappa is None:
                if y.ndim is 3:
                    self.allocate(y.shape[0], cost_model.alssm.N, y.shape[2])
                else:
                    self.allocate(y.shape[0], cost_model.alssm.N)

            # cost_model is CostSegment
            if betas is None:
                betas = 1

            if cost_model.alssm.isUpdated is False:
                cost_model.alssm.update()

            if cost_model.segment.direction is lm.BACKWARD:
                self._backward_recursion(cost_model.alssm, cost_model.segment, y, betas)
            elif cost_model.segment.direction is lm.FORWARD:
                self._forward_recursion(cost_model.alssm, cost_model.segment, y, betas)
            else:
                ValueError("Segment of cost_model has unknown direction.")

    def minimize(self):
        r"""Squared error minimization with linear constraints and offset.

        Minimizes the squared error over the state vector `x`, applying linear constraints with an (optional) offset.
        [Wildhaber2018]_ [TABLE V].

        **Linear constraint:**
        :math:`x=Hv +h,\,v\in\mathbb{M}`
        with :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

        Returns
        -------
        v : :class:`~numpy-ndarra`
            shape(`K`, [`M`,[ `S`]]),
            estimate of the independent states, minimizing the squared  error in the :code:`CCost` with `x=Hv+h`.



       --

       | |def_K|
       | |def_L|
       | |def_M_indep|
       | |def_N|
       | |def_S|
       | |def_P|

       """

        if self.xi.ndim is 2:
            return np.squeeze(np.linalg.pinv(self.W) @ self.xi[:, :, np.newaxis])
        return np.linalg.pinv(self.W) @ self.xi

    def minimize_lin(self, H, h=None):
        r"""Minimization of the cost function with constraints

        Calculation of the state space vector and the constrained state space vector as writen in [Wildhaber2018]_ [
        TABLE V].

        **Constraint:**

        - *Linear Scalar* : :math:`x=Hv,\,v\in\mathbb{R}`

          known : :math:`H \in \mathbb{R}^{N \times 1}`

          :math:`\hat{v}_k = \frac{\xi_k^{\mathsf{T}}H}{H^{\mathsf{T}}W_k H}`

          (if `H=array_like`, `h=None`)

        - *Linear Combination With Offset* : :math:`x=Hv +h,\,v\in\mathbb{M}`

          known : :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

          :math:`\hat{v}_k = \big(H^{\mathsf{T}}W_k H\big)^{-1} H^\mathsf{T}\big(\xi_k - W_k h\big)`

          (if `H=array_like`, `h=array_like`)

        Parameters
        ----------
        H : array_like
            Constrain matrix :math:`H \in \mathbb{R}^{N \times M}`
        h : array_like
            Offset vector :math:`h \in \mathbb{R}^{N}`

        Returns
        -------
        x : :class:`~numpy.ndarray`
            Least square estimate :math:`\hat{x}`
        """
        if self.xi.ndim is 2:
            return np.squeeze(H @ self.minimize_lin_v(H, h=h))
        return H @ self.minimize_lin_v(H, h=h)

    def minimize_lin_v(self, H, h=None):
        if h is not None:
            if self.xi.ndim is 2:
                return (
                    np.linalg.pinv(np.transpose(H) @ self.W @ H)
                    @ np.transpose(H)
                    @ (self.xi[:, :, np.newaxis] - self.W @ h)
                ) + h
            return (
                np.linalg.pinv(np.transpose(H) @ self.W @ H)
                @ np.transpose(H)
                @ (self.xi - self.W @ h)
            ) + h
        if np.shape(H)[1] == 1:
            if self.xi.ndim is 2:
                return (
                    H
                    @ np.transpose(self.xi[:, :, np.newaxis])
                    @ H
                    / (np.transpose(H) @ self.W @ H)
                )
            return H @ np.transpose(self.xi) @ H / (np.transpose(H) @ self.W @ H)
        else:
            if self.xi.ndim is 2:
                return (
                    np.linalg.pinv(np.transpose(H) @ self.W @ H)
                    @ np.transpose(H)
                    @ self.xi[:, :, np.newaxis]
                )
            return (
                np.linalg.pinv(np.transpose(H) @ self.W @ H) @ np.transpose(H) @ self.xi
            )

    def eval_se(self, xs, ref=()):
        """Evaluates the squared error for the current model `CCost` at one or multiple time indices.

        Evaluation of squared error cost at time index `k_i`, where `k_i` is the `i`-th element out of the list
        given by `ks`.
        For evaluation, the corresponding initial state provided by `x[i]` is used.

        Parameters
        ----------
        xs : :class:`numpy.ndarray`
            shape(`K`, [`N` [, `S`]]),
            Initial state space vector(s).

        ref : tuple
            ref is a tuple of two elements. The first element is a list of indices, the second the number of samples


        Returns
        -------
        J : :class:`numpy.ndarray`
            shape(`K` [, `S`]).
            Squared error cost vector of full length `K`. The cost is only evaluated at the time indices specified by
            the list `k`, and set to `nan` otherwise.

        ---

        | |def_K|
        | |def_L|
        | |def_N|
        | |def_S|
        | |def_P|

        """

        if len(ref) is 0:
            ref = (np.arange(len(xs)), len(xs))

        ks = ref[0]
        K = ref[1]

        if np.ndim(xs) <= 2:
            J = np.full((K,), np.nan, dtype=float)
            for k, x in zip(ks, xs):
                J[k] = (
                    np.linalg.multi_dot((x.T, self.W[k], x))
                    - 2 * self.xi[k].T @ x
                    + self.kappa[k]
                )
        else:
            S = np.shape(xs)[2]
            J = np.full((K, S), np.nan, dtype=float)
            for k, x in zip(ks, xs):
                J[k] = np.diag(
                    np.linalg.multi_dot((x.T, self.W[k], x))
                    - 2 * self.xi[k].T @ x
                    + self.kappa[k]
                )
        return J

    def _backward_recursion(self, alssm, seg, y, beta):
        W = np.zeros_like(self.W[0])
        xi = np.zeros_like(self.xi[0])
        kappa = np.zeros_like(self.kappa[0])

        K = self.W.shape[0]

        # set boundaries for backward recursions
        k_first = 1
        k_last = max(K - seg.a, K) + 1

        # Pre calculations
        gamma_a = np.float_power(seg.gamma, seg.a - seg.delta)
        Aa = np.linalg.matrix_power(alssm.A, seg.a)
        Aac = np.dot(Aa.T, alssm.C.T)
        AaccAa = np.linalg.multi_dot(
            (Aa.T, np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C), Aa)
        )
        gamma_b = np.float_power(seg.gamma, seg.b - seg.delta + 1)
        if np.isinf(seg.b):
            Abc = alssm.C.T
            AbccAb = np.dot(np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C))
        else:
            Ab = np.linalg.matrix_power(alssm.A, seg.b + 1)
            Abc = np.dot(Ab.T, alssm.C.T)
            AbccAb = np.linalg.multi_dot(
                (Ab.T, np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C), Ab)
            )

        # iterate
        for k in np.arange(k_last, k_first + 1 - 1, -1):
            W = seg.gamma * (alssm.A.T @ W @ alssm.A)
            xi = seg.gamma * (alssm.A.T @ xi)
            kappa = seg.gamma * kappa

            if MATH_START_INDEX <= k + seg.a - 1 <= K:
                W += gamma_a * AaccAa
                xi += gamma_a * np.dot(Aac, y[k + seg.a - 1 + MATH_TO_LANG_OFFSET])
                kappa += gamma_a * np.dot(
                    y[k + seg.a - 1 + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.a - 1 + MATH_TO_LANG_OFFSET],
                )

            if MATH_START_INDEX <= k + seg.b <= K:
                W -= gamma_b * AbccAb
                xi -= gamma_b * np.dot(Abc, y[k + seg.b + MATH_TO_LANG_OFFSET])
                kappa -= gamma_b * np.dot(
                    y[k + seg.b + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.b + MATH_TO_LANG_OFFSET],
                )

            if MATH_START_INDEX <= k - 1 <= K:
                self.W[k - 1 + MATH_TO_LANG_OFFSET] += W * beta
                self.xi[k - 1 + MATH_TO_LANG_OFFSET] += xi * beta
                self.kappa[k - 1 + MATH_TO_LANG_OFFSET] += kappa * beta

    def _forward_recursion(self, alssm, seg, y, beta):

        W = np.zeros_like(self.W[0])
        xi = np.zeros_like(self.xi[0])
        kappa = np.zeros_like(self.kappa[0])

        K = self.W.shape[0]

        k_first = min(MATH_START_INDEX - seg.b, MATH_START_INDEX) - 1
        k_last = K

        # Pre calculations
        gamma_inv = seg.gamma ** (-1)
        A_inv = np.linalg.inv(alssm.A)

        gamma_a = np.float_power(seg.gamma, seg.a - seg.delta - 1)
        if np.isinf(seg.a):
            Aac = alssm.C.T
            AaccAa = np.dot(np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C))
        else:
            Aa = np.linalg.matrix_power(alssm.A, seg.a - 1)
            Aac = np.dot(Aa.T, alssm.C.T)
            AaccAa = np.linalg.multi_dot(
                (Aa.T, np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C), Aa)
            )

        gamma_b = np.float_power(seg.gamma, seg.b - seg.delta)
        Ab = np.linalg.matrix_power(alssm.A, seg.b)
        Abc = np.dot(Ab.T, alssm.C.T)
        AbccAb = np.linalg.multi_dot(
            (Ab.T, np.atleast_2d(alssm.C).T, np.atleast_2d(alssm.C), Ab)
        )

        # iterate
        for k in np.arange(k_first, k_last - 1 + 1, 1):
            W = gamma_inv * (A_inv.T @ W @ A_inv)
            xi = gamma_inv * (A_inv.T @ xi)
            kappa = gamma_inv * kappa

            if MATH_START_INDEX <= k + seg.a <= K:
                W -= gamma_a * AaccAa
                xi -= gamma_a * np.dot(Aac, y[k + seg.a + MATH_TO_LANG_OFFSET])
                kappa -= gamma_a * np.dot(
                    y[k + seg.a + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.a + MATH_TO_LANG_OFFSET],
                )

            if MATH_START_INDEX <= k + seg.b + 1 <= K:
                W += gamma_b * AbccAb
                xi += gamma_b * np.dot(Abc, y[k + seg.b + 1 + MATH_TO_LANG_OFFSET])
                kappa += gamma_b * np.dot(
                    y[k + seg.b + 1 + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.b + 1 + MATH_TO_LANG_OFFSET],
                )

            if MATH_START_INDEX <= k + 1 <= K:
                self.W[k + 1 + MATH_TO_LANG_OFFSET] += W * beta
                self.xi[k + 1 + MATH_TO_LANG_OFFSET] += xi * beta
                self.kappa[k + 1 + MATH_TO_LANG_OFFSET] += kappa * beta


class SEParamSteadyState(SEParam):
    def __init__(self, label="label missing"):
        """

        Parameters
        ----------
        label : str
            FParam label
        """
        self.v = None
        super(SEParamSteadyState, self).__init__(label=label)

    def set_steady_state_W(self, W):
        self.W = W

    def allocate(self, K, N, S=None):
        """
        Filter Parameter memory allocation

        Parameters
        ----------
        K : int
            Number of samples
        N : int
            Order of ALSSM
        S : int
            Number of sets

        ---

        | |def_K|
        | |def_N|
        | |def_S|
        """

        self.xi = np.zeros((K, N)) if S is None else np.zeros((K, N, S))
        self.kappa = np.zeros((K,)) if S is None else np.zeros((K, S, S))
        self.v = np.zeros((K,)) if S is None else np.zeros((K, S, S))

    def _backward_recursion(self, alssm, seg, y, beta):
        xi = np.zeros_like(self.xi[0])
        kappa = np.zeros_like(self.kappa[0])
        v = np.zeros_like(self.v[0])
        K = self.xi.shape[0]

        # set boundaries for backward recursions
        k_first = 1
        k_last = max(K - seg.a, K) + 1

        # Pre calculations
        gamma_a = np.float_power(seg.gamma, seg.a - seg.delta)
        Aa = np.linalg.matrix_power(alssm.A, seg.a)
        Aac = np.dot(Aa.T, alssm.C.T)
        gamma_b = np.float_power(seg.gamma, seg.b - seg.delta + 1)
        if np.isinf(seg.b):
            Abc = alssm.C.T
        else:
            Ab = np.linalg.matrix_power(alssm.A, seg.b + 1)
            Abc = np.dot(Ab.T, alssm.C.T)

        # iterate
        for k in np.arange(k_last, k_first + 1 - 1, -1):
            xi = seg.gamma * (alssm.A.T @ xi)
            kappa = seg.gamma * kappa
            v = seg.gamma * v

            if MATH_START_INDEX <= k + seg.a - 1 <= K:
                xi += gamma_a * np.dot(Aac, y[k + seg.a - 1 + MATH_TO_LANG_OFFSET])
                kappa += gamma_a * np.dot(
                    y[k + seg.a - 1 + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.a - 1 + MATH_TO_LANG_OFFSET],
                )
                v += gamma_a

            if MATH_START_INDEX <= k + seg.b <= K:
                xi -= gamma_b * np.dot(Abc, y[k + seg.b + MATH_TO_LANG_OFFSET])
                kappa -= gamma_b * np.dot(
                    y[k + seg.b + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.b + MATH_TO_LANG_OFFSET],
                )
                v -= gamma_b

            if MATH_START_INDEX <= k - 1 <= K:
                self.xi[k - 1 + MATH_TO_LANG_OFFSET] += xi * beta
                self.kappa[k - 1 + MATH_TO_LANG_OFFSET] += kappa * beta
                self.v[k - 1 + MATH_TO_LANG_OFFSET] += v * beta

    def _forward_recursion(self, alssm, seg, y, beta):

        xi = np.zeros_like(self.xi[0])
        kappa = np.zeros_like(self.kappa[0])
        v = np.zeros_like(self.v[0])
        K = self.xi.shape[0]

        k_first = min(MATH_START_INDEX - seg.b, MATH_START_INDEX) - 1
        k_last = K

        # Pre calculations
        gamma_inv = seg.gamma ** (-1)
        A_inv = np.linalg.inv(alssm.A)

        gamma_a = np.float_power(seg.gamma, seg.a - seg.delta - 1)
        if np.isinf(seg.a):
            Aac = alssm.C.T
        else:
            Aa = np.linalg.matrix_power(alssm.A, seg.a - 1)
            Aac = np.dot(Aa.T, alssm.C.T)

        gamma_b = np.float_power(seg.gamma, seg.b - seg.delta)
        Ab = np.linalg.matrix_power(alssm.A, seg.b)
        Abc = np.dot(Ab.T, alssm.C.T)

        # iterate
        for k in np.arange(k_first, k_last - 1 + 1, 1):
            xi = gamma_inv * (A_inv.T @ xi)
            kappa = gamma_inv * kappa
            v = gamma_inv * v

            if MATH_START_INDEX <= k + seg.a <= K:
                xi -= gamma_a * np.dot(Aac, y[k + seg.a + MATH_TO_LANG_OFFSET])
                kappa -= gamma_a * np.dot(
                    y[k + seg.a + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.a + MATH_TO_LANG_OFFSET],
                )
                v -= gamma_a

            if MATH_START_INDEX <= k + seg.b + 1 <= K:
                xi += gamma_b * np.dot(Abc, y[k + seg.b + 1 + MATH_TO_LANG_OFFSET])
                kappa += gamma_b * np.dot(
                    y[k + seg.b + 1 + MATH_TO_LANG_OFFSET].T,
                    y[k + seg.b + 1 + MATH_TO_LANG_OFFSET],
                )
                v += gamma_b

            if MATH_START_INDEX <= k + 1 <= K:
                self.xi[k + 1 + MATH_TO_LANG_OFFSET] += xi * beta
                self.kappa[k + 1 + MATH_TO_LANG_OFFSET] += kappa * beta
                self.v[k + 1 + MATH_TO_LANG_OFFSET] += v * beta

    def minimize(self):
        r"""Squared error minimization with linear constraints and offset.

        Minimizes the squared error over the state vector `x`, applying linear constraints with an (optional) offset.
        [Wildhaber2018]_ [TABLE V].

        **Linear constraint:**
        :math:`x=Hv +h,\,v\in\mathbb{M}`
        with :math:`H \in \mathbb{R}^{N \times M},\,h\in\mathbb{R}^N`

        Returns
        -------
        v : :class:`~numpy-ndarra`
            shape(`K`, [`M`,[ `S`]]),
            estimate of the independent states, minimizing the squared  error in the :code:`CCost` with `x=Hv+h`.



       --

       | |def_K|
       | |def_L|
       | |def_M_indep|
       | |def_N|
       | |def_S|
       | |def_P|

       """

        if self.xi.ndim is 2:
            return np.squeeze(
                np.linalg.pinv(self.W) @ self.xi[:, :, np.newaxis], axis=2
            )
        return np.linalg.pinv(self.W) @ self.xi

    def eval_se(self, xs, ref=()):
        """Evaluates the squared error for the current model `CCost` at one or multiple time indices.

        Evaluation of squared error cost at time index `k_i`, where `k_i` is the `i`-th element out of the list
        given by `ks`.
        For evaluation, the corresponding initial state provided by `x[i]` is used.

        Parameters
        ----------
        xs : array_like
            list of state space vectors
        ref : tuple
            (ks, K), list of indices, number of samples

        Returns
        -------
        J : :class:`numpy.ndarray`
            shape(`K` [, `S`]).
            Squared error cost vector of full length `K`. The cost is only evaluated at the time indices specified by
            the list `k`, and set to `nan` otherwise.

        ---

        | |def_K|
        | |def_L|
        | |def_N|
        | |def_S|
        | |def_P|

        """

        if len(ref) is 0:
            ref = (np.arange(len(xs)), len(xs))

        ks = ref[0]
        K = ref[1]

        if np.ndim(xs) <= 2:
            J = np.full((K,), np.nan, dtype=float)
            for k, x in zip(ks, xs):
                J[k] = (
                    np.linalg.multi_dot((x.T, self.W, x))
                    - 2 * self.xi[k].T @ x
                    + self.kappa[k]
                )
        else:
            S = np.shape(xs)[2]
            J = np.full((K, S), np.nan, dtype=float)
            for k, x in zip(ks, xs):
                J[k] = np.diag(
                    np.linalg.multi_dot((x.T, self.W, x))
                    - 2 * self.xi[k].T @ x
                    + self.kappa[k]
                )
        return J
