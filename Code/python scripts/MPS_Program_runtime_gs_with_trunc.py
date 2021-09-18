from MPSfunc_lib import *

'''calculate run time ground state calculation via TEBD without and with
   truncation depending on the particle number N '''

beta = 1
dt0 = 0.4

start_time = time.time()
#-------------------------------------------------------------------------------

'''with truncation'''
#-------------------------------------------------------------------------------
Nmax = 30
eps_trunc = 1e-4
Dmax = 10

#J/beta = 0.5
#-------------
J = 0.5

for i in range(Nmax):
    conf = ImagiTime_config(N=i, J=J, beta=beta, dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    start_time = time.time()
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    stop_time = time.time()
    time_calc_sec = stop_time-start_time
    time_calc_min = time_calc_sec/60
    print(f"with {i:d} {time_calc_sec:f}")

print("\n")

#J/beta = 0.9
#-------------
J = 0.9

for i in range(Nmax):
    conf = ImagiTime_config(N=i, J=J, beta=beta, dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    start_time = time.time()
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    stop_time = time.time()
    time_calc_sec = stop_time-start_time
    time_calc_min = time_calc_sec/60
    print(f"with1 {i:d} {time_calc_sec:f}")

print("\n")

#J/beta = 2
#-------------
J = 2

for i in range(Nmax):
    conf = ImagiTime_config(N=i, J=J, beta=beta, dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    start_time = time.time()
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    stop_time = time.time()
    time_calc_sec = stop_time-start_time
    time_calc_min = time_calc_sec/60
    print(f"with2 {i:d} {time_calc_sec:f}")

#-------------------------------------------------------------------------------
stop_time = time.time()
time_calc_sec = stop_time-start_time
time_calc_min = time_calc_sec/60
print(f"\nScript execution time: {time_calc_sec: f} s = {time_calc_min: f} min")
