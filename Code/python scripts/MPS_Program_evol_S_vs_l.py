from MPSfunc_lib import *

'''calculate entanglement entropy for classical state |00...00> (no entanglement)
   perform then a time evolution for t=0.5 and compute the entanglement entropy
   afterwards'''

N = 100
J = 1
beta = 1

dt = 0.01
steps = 100

start_time = time.time()
#-------------------------------------------------------------------------------
print("#----------------Starting Parameter----------------------")
print(f"N {N:d}")
print(f"J {J:f}")
print(f"beta {beta:f}")
print("#--------------------------------------------------------")

#generate MPS of classical state mps, e.g. |Psi> = |00...00>
mps = []
A = torch.zeros([1, 1, 2], dtype=torch.cdouble)
A[0,0,0] = 1
for i in range(N):
    mps.append(A)

#entanglement entropy before time evolution
#------------------------------------------
#calculate entanglement entropy for different lengths i
Lmax = N-1
for i in range(Lmax):
    print("val1 %f %f" % (i, calc_entanglement_entropy(i, mps)))

#real-time evolution
#-------------------
mps1 = evolveInTime(mps, J, beta, dt, steps)
print("\nEvolution Time: %3.3f" %(steps*dt))
print("\n")

#entanglement entropy after time evolution
#------------------------------------------
#calculate entanglement entropy for different lengths i
for i in range(Lmax):
    print("val2 %f %f" % (i, calc_entanglement_entropy(i, mps1)))

#-------------------------------------------------------------------------------
stop_time = time.time()
print_execution_time(start_time, stop_time)
