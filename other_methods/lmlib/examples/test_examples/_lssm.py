import lmlib as lm

alssm_p = lm.AlssmPoly(Q=2, label="poly")
alssm_s = lm.AlssmSin(omega=0.5, rho=0.2, label="sin")
alssm = lm.AlssmProd((alssm_s, alssm_p), label="multi")
alssm.update()
print("<--PRINT-->")
print(alssm)
print("\n<--DUMP-->")
print(alssm.dump_tree())

alssm_p = lm.AlssmPoly(Q=2, label="poly")
alssm_s = lm.AlssmSin(omega=0.5, rho=0.2, label="sin")
alssm = lm.AlssmProd((alssm_s, alssm_p), G=[5, 0.1], label="multi")
alssm.update()
print(alssm)
print(alssm.dump_tree())


gamma = 0.8
alssm = lm.AlssmExp(gamma)
alssm.update()
print(alssm)

Q = 3
B = [0, 0, 0, 1]
C = [1, 0, 0, 0]
D = 0
lssm = lm.LssmPoly(Q, B, C, D)
lssm.update()
print(lssm)

omega = 0.1
rho = 0.9
B = [1, 1]
C = [0, 1]
D = 0.1
lssm = lm.LssmSin(omega, rho, B, C, D)
lssm.update()
print(lssm)

gamma = 0.8
B = [1]
C = [0.9]
D = 0.1
lssm = lm.LssmExp(gamma, B, C, D)
lssm.update()
print(lssm)
