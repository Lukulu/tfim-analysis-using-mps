from MPSfunc_lib import *

'''calculate one point correlation function expectation value for the ground state
   for different parameter J/beta'''

def one_point_corr_mpo(N):
    B = torch.zeros([3,3, 4], dtype=torch.cdouble)
    #B00:
    B[0,0,0] = 1
    B[2,2,0] = 1
    #B01:
    B[0,1,1] = 1
    B[1,2,1] = 1
    #B10:
    B[0,1,2] = 1
    B[1,2,2] = 1
    #B11:
    B[0,0,3] = 1
    B[2,2,3] = 1

    #left boundary vector
    Brows = B.size()[0]
    Bcols = B.size()[1]
    vL = torch.zeros([1,Brows], dtype=torch.cdouble)
    vL[0,0] = 1.
    #right boundary vector
    vR = torch.zeros([Bcols,1], dtype=torch.cdouble)
    vR[(Bcols-1),0] = 1.

    mpo = []
    mpo.append(vL)
    for k in range(N):
        mpo.append(B.detach().clone())
    mpo.append(vR)
    return mpo

#calculate expectation value MPO
def get_exp_val_mpo(mps, mpo):
    #particle number N
    N = len(mps)

    #intialize result matrix with identity matrix
    prod = torch.zeros([3,3], dtype=torch.cdouble)
    for i in range(3):
         prod[i,i] = 1

    #calculate product of transfer matrices T
    for i in range(N):
        for j in range(2):
            for k in range(2):
                a = torch.conj(mps[i][:,:,j])
                b = mpo[i+1][:,:,getDecRep([j,k])]
                c = mps[i][:,:,k]
                if(j==0 and k==0):
                    T = kronProd(a, kronProd(b,c))
                else:
                    T = torch.add(T, kronProd(a, kronProd(b,c)))
        prod = torch.matmul(prod, T)

    #multiply product of transfer matrices prod with boundary vectors
    res = torch.matmul(mpo[0], torch.matmul(prod, mpo[N+1]))

    #check whether imaginary part is small enough (expect result to be real)
    if (res.imag.abs() > 1e-10):
        print("Warning: Imaginary part > 1e-10")

    return(res.real.numpy()[0])

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

#parameters
#----------
J = np.linspace(0, 4, 30)
beta = np.full(len(J), 1)

rho = one_point_corr_mpo(N)

#ground state from TEBD
#step size adaption
dt0 = 0.4
eps_dt = 0.09

dt_red = 0.8
eps_tol = 1e-3
phys_gap = 1e-3

#truncation
eps_trunc = 1e-4
Dmax = 10
bdim_start = 2

for j in range(len(J)):
    x = J[j]/beta[j]
    conf = ImagiTime_config(N=N, J=J[j], beta=beta[j])
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    E0, E, groundstate = getGroundstate(conf, time_stamp)
    print("val %f %f" % (x, get_exp_val_mpo(groundstate, rho)/(-1)/(N-1)))
