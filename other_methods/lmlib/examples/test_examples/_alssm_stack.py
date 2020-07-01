import lmlib as lm

alssm1 = lm.AlssmPoly(Q=5)
alssm2 = lm.AlssmSin(omega=0.1, rho=0.3)
alssm3 = lm.AlssmPoly(Q=5, C=[[1, 0, 0, 0, 0, 0]])
alssm3.update()
print(alssm3)

alssm_stacked = lm.AlssmStacked((alssm1, alssm2))
alssm_stacked.update()
print(alssm_stacked)

G = [20, 24]

alssm_stacked2 = lm.AlssmStackedSO((alssm1, alssm2), G)
alssm_stacked2.update()
print(alssm_stacked2)

alssm_stacked3 = lm.AlssmStackedSO((alssm3, alssm_stacked2), G=G)
alssm_stacked3.update()
print(alssm_stacked3)

print(alssm_stacked3.C)


print(alssm_stacked3.dump_tree())
