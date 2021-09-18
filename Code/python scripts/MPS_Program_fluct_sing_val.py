from MPSfunc_lib import *

'''examine fluctuations in entanglement entropy for N particles and L=N/2'''

N=50
J=3
beta=1

L=25
niterarions = 100

for i in range(niterarions):
    conf = ImagiTime_config(N=N, J=J, beta=beta)
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    s = get_sing_vals(int(N/2+1), groundstate)
    #print logarithm of first (greatest) singular value
    print("val1 %f\n" % (np.log(s)[0]))
