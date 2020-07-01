import numpy as np
import lmlib as lm

seg = lm.Seg(-20, 20, 30, lm.BACKWARD)
print(seg)

k = [10, 20]
K = 30
window = seg.window(k, K)
print(window)

window_msk = seg.window_msk(k, K, thd=0.7, on_val=200, off_val=np.nan)
print(window_msk)
