"""
Evaluation of Multivariate Polynomials
======================================

"""
import numpy as np
import lmlib as lm

expos = ([0, 2, 4],)
coefs = ([-0.3, 0.08, -0.004],)
m_poly = lm.MPoly(coefs, expos)

print(m_poly.eval((3,)))

print(m_poly.eval((np.arange(5),)))

expos = ([0, 1, 2], [0, 2, 4])
coefs = ([0.1, -0.03, 0.01], [-0.3, 0.08, -0.004])
m_poly = lm.MPoly(coefs, expos)
print(m_poly.eval((1, 3)))

variables = ([1, 2, 3, 4], [3, 2, 1, 0])
print(m_poly.eval(variables))

variables = (
    np.arange(3 * 2 * 5).reshape([3, 2, 5]),
    np.arange(3 * 2 * 5).reshape([3, 2, 5]) - 10,
)
print(m_poly.eval(variables).shape)
