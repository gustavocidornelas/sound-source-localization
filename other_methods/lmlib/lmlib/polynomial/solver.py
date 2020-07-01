from __future__ import division, absolute_import, print_function


__all__ = ["poly_euler_method"]


import numpy as np
import lmlib as lm


def poly_euler_method(poly, x0, step, delta_error):
    """
    Euler method for solving ODEs with given initial value

    .. math::

        f'(x) = f(x)

    Parameters
    ----------
    poly : :class:`~lmlib.polynomial.poly.Poly`
        function
    x0 : scalar
        Initial value
    step : scalar
        Step size
    delta_error : scalar
        termination condition,
        if the error between the current and last value is smaller `delta_error` the iteration stops

    Returns
    -------
    out : scalar
        Returns the x where :math:`f'(x) = f(x)`.

    """
    poly_diff = lm.poly_diff(poly)
    x_old = x0
    x_new = x_old
    y_new = poly.eval(x_new)
    error = np.inf
    while error >= delta_error:
        y_old = y_new
        x_new = x_old + step * poly_diff.eval(x_old)
        y_new = poly.eval(x_new)
        error = abs(y_old - y_new)
    return x_new
