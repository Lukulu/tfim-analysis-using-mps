from MPSfunc_lib import *

'''calculate entangelement behaviour before, close to and after phase transition
   after time evolution'''

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

# beta1 = parameter for calculation of ground state
# beta2 = parameter for calculation of time evolution
def entangelment_behaviour(N, J, beta, prev_gs, use_prev_gs=False):
    #calculate ground state
    conf = ImagiTime_config(N=N, J=J, beta=beta)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    # E0, E, groundstate = getGroundstate(conf, time_stamp)

    #calculate ground state
    if(use_prev_gs):
        E0, E, groundstate = getGroundstate(conf, time_stamp, prev_gs)
    else:
        #random normalized MPS to start iteration
        start = genMPS(gen_bonddims(conf.N, conf.bdim_start))
        right_normalizeMPS(start)
        E0, E, groundstate = getGroundstate(conf, time_stamp, start)

    #entanglement entropy before time evolution depending on subsystem size L
    Lmax = N-1
    entropy_bef = []
    for i in range(Lmax):
        entropy_bef.append(calc_entanglement_entropy(i, groundstate))

    #perform time evolution
    dt = 0.01
    steps = 50

    new_state = evolveInTime(groundstate, J*2, beta, dt, steps)
    print("Evolution Time: %3.3f" %(steps*dt))

    #entanglement entropy after time evolution
    #calculate entanglement entropy for different lengths i
    entropy_aft = []
    for i in range(Lmax):
        entropy_aft.append(calc_entanglement_entropy(i, new_state))

    return entropy_bef, entropy_aft, groundstate;

N = 50
J = np.array([0.1, 1, 1.5])
beta = np.full(len(J), 1)

start_time = time.time()
#-------------------------------------------------------------------------------
#investigate behaviour entanglement entropy before quantum phase transition
entropy_bef1, entropy_aft1, gs1 = entangelment_behaviour(N, J[0], beta[0], 0, False)

#investigate behaviour entanglement entropy after quantum phase transition
entropy_bef2, entropy_aft2, gs2 = entangelment_behaviour(N, J[1], beta[1], 0, False)

#investigate behaviour entanglement entropy at quantum phase transition
entropy_bef3, entropy_aft3, gs3 = entangelment_behaviour(N, J[2], beta[2], gs2, True)

for i in range(len(entropy_bef1)):
    print("val1 %f %f" % (i, entropy_bef1[i]))

print("\n")

for i in range(len(entropy_aft1)):
    print("val2 %f %f" % (i, entropy_aft1[i]))

print("\n")

for i in range(len(entropy_bef2)):
    print("val3 %f %f" % (i, entropy_bef2[i]))

print("\n")

for i in range(len(entropy_aft2)):
    print("val4 %f %f" % (i, entropy_aft2[i]))

print("\n")

for i in range(len(entropy_bef3)):
    print("val5 %f %f" % (i, entropy_bef3[i]))

print("\n")

for i in range(len(entropy_aft3)):
    print("val6 %f %f" % (i, entropy_aft3[i]))

#-------------------------------------------------------------------------------
stop_time = time.time()
print_execution_time(start_time, stop_time)
