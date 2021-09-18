from MPSfunc_lib_new import *

'''calculate E0 (ground state energy) depending on the
   truncation threshold eps for N particles for different J/beta'''

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

Dmax = 1000000
eps =  [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

start_time = time.time()
#-------------------------------------------------------------------------------

for i in range(len(eps)):
    conf = ImagiTime_config(N=N, J=J, beta=beta, dt0=dt0, eps_trunc=eps[i], Dmax=Dmax)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    print("val %f %30.25f" % (eps[i], E))

#-------------------------------------------------------------------------------
stop_time = time.time()
time_calc_sec = stop_time-start_time
time_calc_min = time_calc_sec/60
print(f"\nScript execution time: {time_calc_sec: f} s = {time_calc_min: f} min")
