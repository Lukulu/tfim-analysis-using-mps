from MPSfunc_lib import *

'''calculate ground state energy for fixed step size dt0 depending on dt0
   to estimate Trotter error TEBD'''

#changed convergence criterion for constant step size dt=dt0
def getGroundstate(conf):
    #random normalized MPS to start iteration
    start = genMPS(gen_bonddims(conf.N, conf.bdim_start))
    right_normalizeMPS(start)

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

    #apply 2-particle time evolution gates sequentially
    tmp = start;

    #set step size for start
    dt = conf.dt0

    i = 0
    while(np.abs(rel_err) > conf.eps_dt):
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

        #energy expectation value current state
        prev_energy = energy
        energy      = getEnergyExpVal(tmp, hamiltonian)
        E.append(energy)

        if (i != 0):
            #calculate relative error energy per step
            rel_err = (energy - prev_energy)/prev_energy

        #increase iteration step
        i = i + 1

    print(f"Iteration steps: {i}")
    print(f"Energy ground state: {energy:15.10f}")
    return energy, tmp;

N = 50
J = 0.9
beta = 1
dt0 = pow(10, np.linspace(-2, -1, 10))
eps_trunc = 1e-4
Dmax = 10
eps_con = 1e-6 # convergence threshold for iterations without adaptation of step size

start_time = time.time()
#-------------------------------------------------------------------------------

#vary from dt0 0.4 to 0.01 in steps of 0.039
for i in range(len(dt0)):
    conf = ImagiTime_config(N=N, J=J, beta=beta, dt0=dt0[i], eps_trunc=eps_trunc, Dmax=Dmax, eps_dt=eps_con)
    E, groundstate = getGroundstate(conf)
    print("val %f %f" % (dt0[i], E))

#-------------------------------------------------------------------------------
stop_time = time.time()
time_calc_sec = stop_time-start_time
time_calc_min = time_calc_sec/60
print(f"\nScript execution time: {time_calc_sec: f} s = {time_calc_min: f} min")
