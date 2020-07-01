import numpy as np
import lmlib as lm

# Single-Output ALSSM
A = [[1, 1, 1], [0, 1, 2], [0, 0, 1]]
C = [1, 0, 0]
alssm = lm.Alssm(A, C)

xs = np.arange(12).reshape((2, 3, 2))
js = [-1, 0, 1]
s = alssm.eval_at(xs, js)
print(s)


ks = [4, 7]
K = 10
s = alssm.eval_at(xs, js, ref=(ks, K))
print(s)


s = alssm.eval(xs)
print(s)


s = alssm.eval(xs, ref=(ks, K))
print(s)
