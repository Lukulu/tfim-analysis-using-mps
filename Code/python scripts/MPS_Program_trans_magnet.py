from MPSfunc_lib import *
import gc

'''calculate transversal magnetization expectation value for the ground state
   for different parameter J/beta'''

#parameters
#----------
Ns = [70]
print("N: ", Ns)
J = np.linspace(0, 4, 30)
beta = np.full(len(J), 1)

#numerically:
#-------------------------------------------------------------------------------
for i in range(len(Ns)):
    #calculate longitudinal antimagnetization
    M = getMagnet(Ns[i], False)
    for j in range(len(J)):
        x = J[j]/beta[j]
        conf = ImagiTime_config(N=Ns[i], J=J[j], beta=beta[j])
        time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
        E0, E, groundstate = getGroundstate(conf, time_stamp)
        print("val" + str(i) + " %f %f" % (x, getMagnetExpVal(groundstate, M)/Ns[i]*(-1)))
