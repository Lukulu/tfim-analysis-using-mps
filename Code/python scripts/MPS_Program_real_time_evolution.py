from MPSfunc_lib import *

'''calculate the energy expectation value during real-time evolution'''

#parameter
N = 10
J = 1
beta = 1
dt = 0.01
steps = 200

#generate MPS of classical state mps, e.g. |Psi> = |00...00>
start = []
A = torch.zeros([1, 1, 2], dtype=torch.cdouble)
A[0,0,0] = 1
for i in range(N):
    start.append(A)
right_normalizeMPS(start)

#current date stamp
now = datetime.now().strftime("%y%m%d%H%M%S")

#perform real-time evolution
energy, stop = evolveInTimeEval(start, J, beta, dt, steps, eps_trunc=1e-4, Dmax=10)
