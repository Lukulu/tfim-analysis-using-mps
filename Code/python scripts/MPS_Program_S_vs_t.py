from MPSfunc_lib import *

'''calculate entanglement entropy depending on evolution time t starting
   from a classical state'''

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

dt = 0.0001

start_time = time.time()
#-------------------------------------------------------------------------------
#calculate ground state
conf = ImagiTime_config(N=N, J=J, beta=beta)
time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
E0, E, groundstate = getGroundstate(conf, time_stamp)

#evolve ground state
steps = 0
L=int(N/2)

for i in range(20):
    new_mps = evolveInTime(groundstate, J, beta, dt, steps)
    print("val %f %f" % (steps*dt, calc_entanglement_entropy(L, new_mps)))
    steps += 500

#-------------------------------------------------------------------------------
stop_time = time.time()
print_execution_time(start_time, stop_time)
