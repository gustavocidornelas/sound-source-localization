import lmlib as lm

A = [[1, 2], [3, 4]]
B = [1, 2]
C = [[4, 3]]
D = [3]

lssm = lm.Lssm(A, B, C, D)
lssm.update()
print(lssm)

alssm = lm.AlssmSin(omega=0.1, rho=0.3)

lssm_stacked = lm.LssmStacked((lssm, alssm))
lssm_stacked.update()
print(lssm_stacked)
print(lssm_stacked.D)

lssm_stacked_so = lm.LssmStackedSO((lssm, alssm))
lssm_stacked_so.update()
print(lssm_stacked_so)


lssm_stacked_ci = lm.LssmStackedCI((lssm, alssm))
lssm_stacked_ci.update()
print(lssm_stacked_ci.B)


lssm_stacked_ciso = lm.LssmStackedCISO((lssm, alssm))
lssm_stacked_ciso.update()
print(lssm_stacked_ciso.C)
