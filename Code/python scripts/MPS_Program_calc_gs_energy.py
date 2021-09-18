from MPSfunc_lib import *

'''calculate energy expectation value of the ground state depending
   on parameter J/beta for given N'''

# <><><><> ARGUMENT PARSER <><><><><><><><><><><><><>><><>
import argparse
parser = argparse.ArgumentParser(description='calculate entanglement entropy of the ground state depending on parameter beta/J for given L, N')
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

def getGroundstate(conf, date_stamp, start):
    #hamiltonian
    hamiltonian = genH(len(start), conf.J, conf.beta)

    #generate 2-particle gates U1, U2, U3 for start step size dt0
    U1, U2, U3 = gen2particleGatesU(0, conf.J, conf.beta, conf.dt0)

    #energy expectation value per step
    E = []
    energy = getEnergyExpVal(start, hamiltonian)
    E.append(energy)

    #relative error energy per step
    rel_err = 1
    #exponential error
    exp_err = conf.eps_tol

    #slope energy per time
    slope = 1

    #apply 2-particle time evolution gates sequentially
    tmp = start;

    #extrapolation exact ground state energy E0 from trotter error using E1=E(dt1) and E2=E(dt2)
    E1 = energy
    dt1 = conf.dt0 + 0.1

    #Dumping all the output into file.
    #date_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    dump_stream = open(f"./out/dump/run_groundstate_N{conf.N}_J{conf.J}_beta{conf.beta}_{date_stamp}.txt", 'w')
    dump_stream.write(f"{'[conf]':12s} {'N':>10s} {conf.N:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'J':>10s}  {conf.J:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'beta':>10s} {conf.beta:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'dt0':>10s} {conf.dt0:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'eps_trunc':>10s} {conf.eps_trunc:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'eps_dt':>10s} {conf.eps_dt:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'eps_tol':>10s} {conf.eps_tol:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'Dmax':>10s} {conf.Dmax:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'dt_red':>10s} {conf.dt_red:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'phys_gap':>10s} {conf.phys_gap:>15.3e}\n")

    dump_stream.write(f"{'[info]':12s} {'i':>5s} {'MBD':>8s} {'dt':>17s} {'E':>17s} {'rel_err':>17s} {'exp_err':>17s} {'slope':>17s}\n")

    #set step size for start
    dt = conf.dt0

    i = 0
    while(dt > conf.eps_dt):
        #apply 2-particle gates U1, U2, U3
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U1, dt, tmp, j)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)

        #normalization & truncation
        mean_bonddim = truncateMPS(tmp, conf.eps_trunc, conf.Dmax)
        #mean_bonddim = 1

        #-----------------------------------------------------------------------
        #right-normalize MPS
        right_normalizeMPS(tmp)
        #-----------------------------------------------------------------------

        #energy expectation value current state
        prev_energy = energy
        energy      = getEnergyExpVal(tmp, hamiltonian)
        E.append(energy)

        #calculate slope energy vs. t
        slope = (energy-prev_energy)/dt

        if (i != 0):
            #calculate relative error energy per step
            rel_err = (energy - prev_energy)/prev_energy

            #calculate exponential error
            exp_err = np.abs(rel_err)/np.abs(conf.phys_gap)/pow(dt, 4)

        #reduce time step if exp_err < eps_tol by dt_red
        if (exp_err < conf.eps_tol):
            dt_prev = dt
            dt = dt * conf.dt_red
            U1, U2, U3 = gen2particleGatesU(0, conf.J, conf.beta, dt)

            #extrapolation exact ground state energy E0 from trotter error by E1=E(dt1) and E2=E(dt2)
            d1 = pow(dt1, 4)
            d2 = pow(dt_prev, 4)
            E0 = d1*d2/(d2-d1)*(E1/d1 - energy/d2)
            E1 = energy
            dt1 = dt_prev

            #print(f"(i={i:3d}) reduced time step from {dt_prev:10.6f} to {dt:10.6f}")
            dump_stream.write(f"{'[update_dt]':12s} (i={i:3d}) Reduced time step from {dt_prev:10.6f} to {dt:10.6f}\n")
            dump_stream.write(f"{'[update_E0]':12s} (i={i:3d}) Extrapolated energy ground state E0: {E0:10.6f}\n")

        #print("(i=%3d) MBD=%8.3f, E=%15.10f, rel_err=%12.6e, exp_err=%12.6e" %(i, mean_bonddim, energy, rel_err, exp_err))
        dump_stream.write(f"{'[itstep]':12s} {i:5d} {mean_bonddim:8.3f} {dt:17.3e} {energy:17.10f} {rel_err:17.6e} {exp_err:17.6e} {slope:17.6e}\n")

        #increase iteration step
        i = i + 1

    dump_stream.write(f"\n{'[info]':12s} {'steps':>5s} {'dt':>8s} {'E0':>17s} {'E':>17s}\n")
    dump_stream.write(f"{'[output]':12s} {i:5d} {dt:8.3f} {E0:17.10f} {energy:17.10f}\n")

    dump_stream.close()
    return E0, energy, tmp;

#parameter
J = np.linspace(0,2, 20)
beta = np.full(len(J), 1)

for i in range(len(J)):
    conf = ImagiTime_config(N=N, J=J[i], beta=beta[i], dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax, eps_dt=eps_dt, dt_red=dt_red, eps_tol=eps_tol, phys_gap=phys_gap, bdim_start=bdim_start)
    date_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    if(i < 5):
        #random normalized MPS to start iteration
        start = genMPS(gen_bonddims(conf.N, conf.bdim_start))
        right_normalizeMPS(start)
        E0, E, groundstate = getGroundstate(conf, date_stamp, start)
    else:
        E0, E, groundstate = getGroundstate(conf, date_stamp, groundstate)

    x = J[i]/beta[i]
    e_over_n = E/N
    print(f"val {x:f} {e_over_n:f}")
