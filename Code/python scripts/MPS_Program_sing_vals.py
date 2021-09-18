from MPSfunc_lib import *

'''calculate the singular values corresponding to the intersection point k=N/2'''

# <><><><> ARGUMENT PARSER <><><><><><><><><><><><><>><><>
import argparse
parser = argparse.ArgumentParser(description='calculate the entanglement entropy S depending on the subsystem size L for a different number N of particles')
parser.add_argument('-N',         type=int,   default=20,    help='particle number')
parser.add_argument('-J',         type=float, default=-1,    help='nearest neighbour coupling strength')
parser.add_argument('-b',         type=float, default=-10,   help='external field coupling strength')
parser.add_argument('-eps_trunc', type=float, default=1e-5,  help='truncation threshold discarded weights')
parser.add_argument('-Dmax',      type=int,   default=10,    help='maximal bond dimension after truncation')
parser.add_argument('-dt0',       type=float, default=0.4,   help='step size at t=0 (start)')
parser.add_argument('-eps_dt',    type=float, default=0.03,  help='convergence threshold for steps size dt')
parser.add_argument('-dt_red',    type=float, default=0.8,   help='value by which the step size is reduced')
parser.add_argument('-eps_tol',   type=float, default=0.001, help='tolerance threshold of exponential error')
parser.add_argument('-phys_gap',  type=float, default=0.001, help='gap between ground state & 1st excited level')
parser.add_argument('-bdim',      type=int,   default=2,     help='bond dimension at t=0 (start)')

args = parser.parse_args()
if (args.N != None):
    N = args.N
if (args.J != None):
    J = args.J
if (args.b != None):
    beta = args.b
if (args.eps_trunc != None):
    eps_trunc = args.eps_trunc
if (args.Dmax != None):
    Dmax = args.Dmax
if (args.dt0 != None):
    dt0 = args.dt0
if (args.eps_dt != None):
    eps_dt = args.eps_dt
if (args.dt_red != None):
    dt_red = args.dt_red
if (args.eps_tol != None):
    eps_tol = args.eps_tol
if (args.phys_gap != None):
    phys_gap = args.phys_gap
if (args.bdim != None):
    bdim_start = args.bdim

# <><><><> ARGUMENT PARSER END <><><><><><><><><><><><><>

start_time = time.time()
#-------------------------------------------------------------------------------

print("#----------------Starting Parameter----------------------")
print(f"N {N:d}")
print(f"J {J:f}")
print(f"beta {beta:f}")
print(f"eps_trunc {eps_trunc:e}")
print(f"Dmax {Dmax:d}")
print(f"dt0 {dt0:f}")
print(f"eps_dt {eps_dt:f}")
print(f"dt_red {dt_red:f}")
print(f"eps_tol {eps_tol:f}")
print(f"phys_gap {phys_gap:f}")
print(f"bdim_start {bdim_start:d}")
print("#--------------------------------------------------------")

conf = ImagiTime_config(N=N, J=J, beta=beta, dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax, eps_dt=eps_dt, dt_red=dt_red, eps_tol=eps_tol, phys_gap=phys_gap, bdim_start=bdim_start)
date_stamp = datetime.now().strftime("%y%m%d%H%M%S")
E0, E, groundstate = getGroundstate(conf, date_stamp)

s = get_sing_vals(int(N/2+1), groundstate)

for j in range(len(s)):
    print("val1  %f %30.25f" % (j, s[j]))

#-------------------------------------------------------------------------------
stop_time = time.time()
print_execution_time(start_time, stop_time)
