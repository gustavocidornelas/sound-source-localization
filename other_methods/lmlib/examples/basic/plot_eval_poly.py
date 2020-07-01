"""
Evaluation of univariate Polynomials
====================================

"""
import numpy as np
import lmlib as lm

expo = [0, 2, 4]
coef = [-0.3, 0.08, -0.004]
poly = lm.Poly(coef, expo)

print(poly.eval(3))

print(poly.eval(np.arange(5)))

var = np.arange(2 * 4 * 6).reshape([2, 4, 6])
print(poly.eval(var).shape)
