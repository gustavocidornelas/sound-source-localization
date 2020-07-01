import lmlib as lm


poly = lm.AlssmPoly(4, label="polynomial model")

A = [[1, 1], [0, 1]]
C = [1, 0]
line = lm.Alssm(A, C, label="line model")

stacked_alssm = lm.AlssmStacked((poly, line), label="stacked model")
stacked_alssm.update()
print(stacked_alssm.dump_tree())


alssm = lm.AlssmPoly(Q=3)
alssm.update()
str = alssm.dump_tree()
print(str)
