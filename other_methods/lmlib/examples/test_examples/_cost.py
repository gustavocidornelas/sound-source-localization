import lmlib as lm
import numpy as np

alssm = lm.AlssmPoly(2, label="polynomial_alssm")
alssm.update()

seg = lm.Seg(-20, 0, 10, lm.FORWARD)

cost_seg = lm.CostSeg(alssm, seg, label="cost_poly_left")
print(cost_seg)

k = [100, 150]
K = 400
x = [[0, 1, 2], [3, 4, 5]]
traj = cost_seg.trajectory(k, K, x)
print(traj.shape)


segL = lm.Seg(-20, 0, 10, lm.FORWARD)
segR = lm.Seg(1, 20, 10, lm.BACKWARD)

F = [[1], [2]]
c_cost = lm.CCost((alssm,), (segL, segR), F)

win = c_cost.window(k, K)
win_msk = c_cost.window_msk(k, K, thd=1e-6, on_val=True, off_val=np.nan)
print(win.shape, win_msk.shape)

traj = c_cost.trajectory(k, K, x)
print(traj.shape)
