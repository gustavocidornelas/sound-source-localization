import lmlib as lm
import numpy as np

#################################################
# Create FParam with a cost segment
#

alssm = lm.AlssmPoly(2, label="polynomial_alssm")
alssm.update()

seg = lm.Seg(-20, 0, 10, lm.FORWARD)

cost_seg = lm.CostSeg(alssm, seg, label="cost_poly_left")

y = np.sin(2 * np.pi * np.linspace(0, 1, 100))
f_param = lm.FParam()
f_param.filter(cost_seg, y)
x = f_param.minimize()

#################################################
# Create FParam with a composite cost
#

segL = lm.Seg(-20, 0, 10, lm.FORWARD)
segR = lm.Seg(1, 20, 10, lm.BACKWARD)

F = [[1], [2]]
c_cost = lm.CCost((alssm,), (segL, segR), F)

f_param = lm.FParam()
f_param.filter(cost_seg, y)
x2 = f_param.minimize()
