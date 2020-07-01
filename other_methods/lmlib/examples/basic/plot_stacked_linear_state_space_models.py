"""
.. _lmlib_example_stacked_linear_state_space_models:

Stacked Linear State Space Models
=================================

This example generates multiple discrete-time linear state space models and stacks them . :meth:`dump_tree()`
returns the structure of the stacked Lssm.

"""
import lmlib as lm

A = [[1, 1], [0, 1]]
C = [1, 0]
alssm = lm.Alssm(A, C, label="alssm-line")

Q = 1
B = [1, 0]
C = [1, 0]
D = [0]
lssmPoly = lm.LssmPoly(Q, B, C, D, label="lssm-polynomial")

Q = 3
alssmPoly = lm.AlssmPoly(Q, label="alssm-polynomial")

# stacking lssm and update
lssmStacked = lm.LssmStacked((alssmPoly, lssmPoly, alssm), label="lssm-stacked")

# print structure and content
print("--DUMP--\n", lssmStacked.dump_tree())
print("--PRINT--\n", lssmStacked)
