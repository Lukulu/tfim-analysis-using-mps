#!/usr/bin/env python
# coding: utf-8
#@author: Laura Zywietz Rolón

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit
from datetime import datetime #generate time stamp for plotting
import random
import torch #tensor manipulation
import cmath #complex numbers/functions
import math #math.floor
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm #color plots
import time #runtime analysis
import statistics #runtime analysis
#for dataclass
from dataclasses import dataclass
from typing import Any
import gc #for destruction of tensor objects
import copy

'''helper functions'''
#-----------------------------------------------------------------------
#kronecker product of two 2-dim tensors a,b
def kronProd(a,b):
    a_np = a.numpy()
    b_np = b.numpy()
    res_np = np.kron(a_np, b_np)
    res = torch.from_numpy(res_np)
    return(res)

#get decimal respresentation of a binary number bin
def getDecRep(bin):
    a = 0
    for i in range(len(bin)):
        a = a + bin[i]*pow(2, (len(bin)-i-1))
    return(a)

#calculate binary representation of number num with N bits
def getBinaryRep(num, N):
    bin = []
    for i in range(N):
        bin.append(int(math.floor(num/pow(2, (N-1-i))) % 2))
    return bin

#print the execution time of a code block for given start and stop times
def print_execution_time(start_time, stop_time):
    time_calc_sec = stop_time - start_time
    time_calc_min = time_calc_sec/60
    print(f"Execution time: {time_calc_sec: f} s = {time_calc_min: f} min")

@dataclass
class params:
    N: int           #particle number,
    J: float         #nearest neighbour coupling strength,
    beta: float      #external field coupling strength,
    eps_trunc: float #truncation threshold discarded weights,
    Dmax: int        #maximal bond dimension after truncation,
    dt0: float       #step size at t=0 (start),
    eps_dt: float    #convergence threshold for steps size dt,
    dt_red: float    #value by which the step size is reduced,
    eps_tol: float   #tolerance threshold of exponential error,
    phys_gap: float  #gap between ground state & 1st excited level,
    bdim_start: int  #bond dimension at t=0 (start)

#read data from output files fn with keyword kw = "val"
def parseJobOutput(fn, kw):
    # open file and load all lines into memory
    f = open(fn, 'r')
    lines = f.readlines()
    f.close()

    par = params(0, 0., 0., 0., 0, 0., 0., 0., 0., 0., 0)

    for line in lines:
        if (len(line) == 1 and line=="\n"):
            continue

        if (line.split()[0][0] == '#'):
            continue

        var_name = line.split()[0]
        var_val = line.split()[1]
        if (var_name == 'N'):
            par.N = int(var_val)
        if (var_name == 'J'):
            par.J = float(var_val)
        if (var_name == 'beta'):
            par.beta = float(var_val)
        if (var_name == 'eps_trunc'):
            par.eps_trunc = float(var_val)
        if (var_name == 'Dmax'):
            par.Dmax = int(var_val)
        if (var_name == 'dt0'):
            par.dt0 = float(var_val)
        if (var_name == 'eps_dt'):
            par.eps_dt = float(var_val)
        if (var_name == 'dt_red'):
            par.dt_red = float(var_val)
        if (var_name == 'eps_tol'):
            par.eps_tol = float(var_val)
        if (var_name == 'phys_gap'):
            par.phys_gap = float(var_val)
        if (var_name == 'bdim_start'):
            par.bdim_start = int(var_val)

    line_counter = 0
    dim = 0
    for line in lines:
        if (len(line) == 1 and line=="\n"):
            continue
        if (line.split()[0][0] == '#'):
            continue
        if (line.split()[0] == kw):
            line_counter += 1
            dim = len(line.split()) - 1

    data = np.zeros((dim, line_counter))
    k = 0
    for line in lines:
        if (len(line) == 1 and line=="\n"):
            continue
        if (line.split()[0][0] == '#'):
            continue
        if (line.split()[0] == kw):
            for i in range(dim):
                data[i, k] = float(line.split()[i+1])
            k += 1
    return (par, data)
#-----------------------------------------------------------------------

'''functions for plotting'''
#-----------------------------------------------------------------------
#create canvas with figuresize: fs, figureproportions: fp
def mp_canvas(fs, fp):
    figproportions = fp
    figsize = fs
    plt.rcParams.update({'font.size': 25})

    fig     = plt.figure(datetime.now().strftime("%H%M%S"), figsize=(figsize * figproportions, figsize))
    ax      = fig.add_subplot()
    ax.grid()
    ax.set_axisbelow(True)
    return (fig, ax)

#save figure fig to PDF file with filename fn
def mp_savePDF(fn, fig):
    now = datetime.now().strftime("%y%m%d%H%M%S")
    fn = "./out/%s_%s.pdf" % (fn, now)
    #fn = "./out/%s.pdf" % (fn)
    fig.savefig(fn, bbox_inches='tight')

#load data from textfile df
def mp_loadtxtfile(df):
    return np.loadtxt(df, unpack=True)

#plot errors
def mp_errorbar(ax, x, y, dy, lab):
    ax.errorbar(x=x, y=y, yerr=dy,
        color='blue',
        label=lab,
        marker="o", linestyle='', markersize=2,
        elinewidth=1, capsize=2
        )

#fit function f to data x, y
def mp_curveFit(x, y, dy, f, popt):
    popt, pcov = curve_fit(f, x, y, sigma=dy, p0=popt)
    print("Fit parameter: ")
    for i in range(0, len(popt)):
        print("%3d: %10.4f +- %10.4f" %(i, popt[i], math.sqrt(pcov[i,i])))
    return (popt, pcov)

#calculate chi squared of  fit
def chisq(obs, exp, error):
    return np.sum((obs - exp) ** 2 / (error ** 2))

def mp_chisq(x, y, dy, f, popt):
    chis = chisq(f(x, *popt), y, dy)
    dof   = len(x)-len(popt)-1

    print("Chisq / d.o.f:")
    print("%10.4f / %5.0f" %(chis, dof))
    return (chis, dof)
#-----------------------------------------------------------------------

'''MPS functions'''
#-----------------------------------------------------------------------
#generate bond dimensions mps for N particles from bdim
def gen_bonddims(N, bdim):
    bonddims = []

    if(type(bdim) is int):
        bonddims.append([1,bdim, 2])
        for i in range(1, N-1):
            bonddims.append([bdim,bdim, 2])
        bonddims.append([bdim,1, 2])
    else:
        assert(len(bdim) == N-1), "Incorrect dimension input!"
        bonddims.append([1,bdim[0], 2])
        for i in range(1, N-1):
            bonddims.append([bdim[i-1],bdim[i], 2])
        bonddims.append([bdim[N-2],1, 2])

    return(bonddims)

#generate random unnormalized MPS
def genMPS(bonddim):
    #particle number N
    N = len(bonddim)

    mps = []
    for i in range(N):
        #check for correct dimensions for the cotnracted indices
        if (i < N-1):
            assert (bonddim[i+1][0] == bonddim[i][1]), "Error: Subsequent indices should have the same dimension!"
        mps.append(torch.ones(bonddim[i], dtype=torch.cdouble).uniform_(-1,1))
    return(mps)

#print MPS
def printMPS(mps):
    print("----------MPS elements-----------")
    for i in range(len(mps)):
        print("===========particle ", i, " ===========")
        print("---------------s=0---------------")
        print(mps[i][:,:,0])
        print(" ")
        print("---------------s=1---------------")
        print(mps[i][:,:,1])
        print(" ")
        print("=================================")

#compute MPS elements from given coeff of the form of
#2x2**(N-1) matrix through sucessive SVD's
def genMPS_from_coeff(coeff):
    assert (coeff.size()[0] == 2),"Error: Incorrect input form!"

    #particle number N
    N = round(np.log2(coeff.size()[0]*coeff.size()[1]))
    print("N = ", np.log2(coeff.size()[0]*coeff.size()[1]))

    tmp = coeff

    mps = []
    for i in range(N-1):
        #SVD
        u, s, vh = np.linalg.svd(tmp.numpy(), full_matrices=False)
        num_s = len(s)
        #calculate s * vh
        svh = np.matmul(np.diag(s), vh)

        #recasting objects from numpy arrays to pytorch tensors
        ut = torch.from_numpy(u)
        svht = torch.from_numpy(svh)
        vht = torch.from_numpy(vh)

        #MPS tensor elements, obtained by reshaping the matrix u according to the prescription above:
        #(a, s) = 2*a + s
        nrows = round(ut.size()[0]/2)
        ncols = ut.size()[1]
        A = torch.zeros([nrows, ncols, 2], dtype=torch.cdouble)
        for j in range(nrows):
            A[j,:,0] = ut[2*j,:]
            A[j,:,1] = ut[2*j+1,:]
        mps.append(A)

        #check whether SVD is going to continue or the last step is reached, where the last
        #tensor component of the MPS is given by C**(N-1) = s*(N-2)*vh**(N-2)
        if(i != N-2):
            #for the next SVD the product s*vh is transformed from a axb matrix to a (2*a)x(b/2) matrix
            #according to the prescription (a,s) = 2*a + s (for the rows)
            rows = svht.size()[0]
            cols = svht.size()[1]
            nextCoeff = torch.zeros([rows*2, round(cols/2)], dtype=torch.cdouble)
            s = -1
            for j in range(rows):
                for k in range(cols):
                    if(k == nextCoeff.size()[1] or k == 0):
                        s = s+1
                    nextCoeff[s, k%(nextCoeff.size()[1])] = svht[j,k]
            tmp = nextCoeff
        else:
            #calculate last tensor component of MPS from the product s*vh
            #from the full matrix s, vh, which is obtained by filling additional entries
            #with zero, corresponding to the multiplication with singular values = 0
            fullsvht = torch.zeros([ut.size()[1], vht.size()[1]], dtype=torch.cdouble)
            for j in range(num_s):
                fullsvht[j, :] = svht[j,:]
            lastA = torch.zeros([fullsvht.size()[0], 1, 2], dtype=torch.cdouble)
            lastA[:,0,0] = fullsvht[:,0]
            lastA[:,0,1] = fullsvht[:,1]
            mps.append(lastA)
    return(mps)

#calculate coeffcients state wrt to computational basis
#from a given MPS mps
def calc_coeff_state(mps):
    N = len(mps)
    num_coeff = pow(2, N)
    coeff = torch.zeros([num_coeff,1], dtype= torch.cdouble)
    for i in range(num_coeff):
        ind = getBinaryRep(i, N)
        prod = mps[0][:,:,ind[N-1]]
        for j in range(1,N):
            prod = torch.matmul(prod, mps[j][:,:,ind[(N-1)-j]])
        coeff[i,0] = prod
    return(coeff)
#-----------------------------------------------------------------------

'''MPO functions'''
#-----------------------------------------------------------------------
#generate arbitrary MPO from tensor B
def genMPO(length, B):
    #boundary vectors
    #left
    Brows = B.size()[0]
    Bcols = B.size()[1]
    vL = torch.zeros([1,Brows], dtype=torch.cdouble)
    vL[0,0] = 1.
    #right
    vR = torch.zeros([Bcols,1], dtype=torch.cdouble)
    vR[(Bcols-1),0] = 1.

    mpo = []
    mpo.append(vL)
    for i in range(length):
        mpo.append(B)
    mpo.append(vR)

    return(mpo)

#print MPO
def printMPO(length, mpo):
    print("----------MPO elements-----------")
    print("==============vL=================")
    print(mpo[0])
    print(" ")
    print("=================================")
    for j in range(1,(length+1)):
        print("=================================")
        print("---------------s=00---------------")
        print(mpo[j][:,:,0])
        print(" ")
        print("---------------s=01---------------")
        print(mpo[j][:,:,1])
        print(" ")
        print("---------------s=10---------------")
        print(mpo[j][:,:,2])
        print(" ")
        print("---------------s=11---------------")
        print(mpo[j][:,:,3])
        print(" ")
        print("=================================")
    print("==============vR=================")
    print(mpo[length+1])
    print(" ")
    print("=================================")

#generate Hamiltonian as MPO
def genH(length, J, beta):
    #generate tensor B
    B = torch.zeros([3,3,4], dtype=torch.cdouble)
    #B_0
    B[0,0,0] = 1.
    B[2,2,0] = 1.
    B[0,2,0] = beta
    #B_1
    B[0,1,1] = J
    B[1,2,1] = 1.
    #B_2
    B[:,:,2] = B[:,:,1]
    #B_3
    B[0,0,3] = 1.
    B[2,2,3] = 1.
    B[0,2,3] = -1*beta

    #boundary vectors
    #left
    vL = torch.zeros([1,3], dtype=torch.cdouble)
    vL[0,0] = 1.
    #right
    vR = torch.zeros([3,1], dtype=torch.cdouble)
    vR[2,0] = 1.

    hamiltonian = []
    hamiltonian.append(vL)
    for i in range(length):
        hamiltonian.append(B)
    hamiltonian.append(vR)

    return(hamiltonian)
#-----------------------------------------------------------------------

'''MPO-MPS functions'''
#-----------------------------------------------------------------------
#left gauging:
def left_gauge(pos, mps):
    assert(pos >= 0 and pos < len(mps)-1), "Given position exceeds length of MPS!"

    #reshaping of tensor mps[pos] and mps[pos+1] into matrix (a_i,s_i) x a_i+1 and a_i+1 x (a_i+2, s_i+1)
    ai = mps[pos].size()[0]
    ai1 = mps[pos].size()[1]
    ai2 = mps[pos+1].size()[1]

    A = torch.zeros([2*ai, ai1], dtype=torch.cdouble)
    for j in range(ai):
        A[2*j,:] = mps[pos][j,:,0]
        A[2*j+1, :] = mps[pos][j,:,1]

    C = torch.zeros([ai1, 2*ai2], dtype=torch.cdouble)
    for j in range(ai2):
        C[:, 2*j] = mps[pos+1][:,j,0]
        C[:,2*j+1] = mps[pos+1][:,j,1]

    #SVD
    u, s, vh = np.linalg.svd(A.numpy(), full_matrices=False)

    #recasting objects from numpy arrays to pytorch tensors
    ut = torch.from_numpy(u)
    st = torch.from_numpy(s)
    vht = torch.from_numpy(vh)

    #get matrix s*v*C using complex matrix multiplication of numpy
    sv_tmp = np.matmul(torch.diag(st).numpy(), vht.numpy())
    svc_n = np.matmul(sv_tmp, C.numpy())
    svc = torch.from_numpy(svc_n)

    #number of singular values
    num_s = st.size()[0]

    #define new MPS tensor elements
    A_tilde = torch.zeros([ai, num_s, 2], dtype=torch.cdouble)
    nextA_tilde =torch.zeros([num_s, ai2, 2], dtype=torch.cdouble)

    #reshaping ut into new MPS tensors with the given prescription
    for j in range(ai):
        A_tilde[j,:,0] = ut[2*j,:]
        A_tilde[j,:,1] = ut[2*j+1,:]

    for j in range(ai2):
        nextA_tilde[:,j,0] = svc[:,2*j]
        nextA_tilde[:,j,1] = svc[:,2*j+1]

    #set new MPS elements
    mps[pos] = A_tilde
    mps[pos+1] = nextA_tilde

#right gauging:
def right_gauge(pos, mps):
    assert(pos > 0 and pos < len(mps)), "Given position exceeds length of MPS!"

    #reshaping of tensor mps[pos-1] and mps[pos] into matrix (s_i-1,a_i-1) x a_i and a_i x (s_i, a_i+1)
    ai0 = mps[pos-1].size()[0]
    ai = mps[pos].size()[0]
    ai1 = mps[pos].size()[1]

    A = torch.zeros([ai, 2*ai1], dtype=torch.cdouble)
    for j in range(2*ai1):
        if(j < ai1):
            A[:,j] = mps[pos][:,j, 0]
        else:
            A[:,j] = mps[pos][:,j-ai1, 1]

    C = torch.zeros([2*ai0, ai], dtype=torch.cdouble)
    for j in range(2*ai0):
        if(j < ai0):
            C[j,:] = mps[pos-1][j,:, 0]
        else:
            C[j, :] = mps[pos-1][j-ai0,:, 1]

    #SVD
    u, s, vh = np.linalg.svd(A.numpy(), full_matrices=False)

    #recasting objects from numpy arrays to pytorch tensors
    ut = torch.from_numpy(u)
    st = torch.from_numpy(s)
    vht = torch.from_numpy(vh)

    #get matrix C*u*s using complex matrix multiplication of numpy
    us_tmp = np.matmul(ut.numpy(), torch.diag(st).numpy())
    cus_n = np.matmul(C.numpy(), us_tmp)
    cus = torch.from_numpy(cus_n)

    #number of singular values
    num_s = st.size()[0]

    #define new MPS tensor elements
    A_tilde = torch.zeros([num_s, ai1, 2], dtype=torch.cdouble)
    prevA_tilde =torch.zeros([ai0, num_s, 2], dtype=torch.cdouble)

    #reshaping ut into new MPS tensors with the given prescription
    for j in range(2*ai1):
        if(j < ai1):
            A_tilde[:,j,0] = vht[:,j]
        else:
            A_tilde[:,j-ai1,1] = vht[:,j]

    for j in range(2*ai0):
        if(j < ai0):
            prevA_tilde[j,:,0] = cus[j,:]
        else:
            prevA_tilde[j-ai0,:,1] = cus[j,:]

    #set new MPS elements
    mps[pos] = A_tilde
    mps[pos-1] = prevA_tilde

#obtain central gauge of MPS mps with gauge center at site k+1
def central_gauge(k, mps):
    N = len(mps)
    assert(k >=0 and k < N), "Given position exceeds length of MPS!"

    #left gauge all MPS elements to the left of site k, including k
    for i in range(k+1):
        left_gauge(i, mps)
    #right gauge all MPS elements to the right of site k, starting from k+2
    for i in range(N-1, k+1, -1):
        right_gauge(i, mps)

#normalization MPS through left and right gauging
def left_normalizeMPS(mps):
    length = len(mps)

    #left-gauge whole MPS up to the last element
    for i in range(length-1):
        left_gauge(i, mps)

    #calculate normalization constant C
    C = torch.zeros([1,1], dtype=torch.cdouble)
    for i in range(2):
        prod = torch.matmul(torch.transpose(torch.conj(mps[length-1][:,:,i]), 0,1), mps[length-1][:,:,i])
        C = C + prod

    #normalize mps by adapting last mps element
    normConst = pow(C, 1./2)
    mps[length-1] = mps[length-1]*(1./normConst)

    #return normalization constant, e.g. for debugging
    return(np.abs(C).item())

def right_normalizeMPS(mps):
    length = len(mps)

    #right-gauge whole MPS up the first element
    for i in range(length-1, 0, -1):
        right_gauge(i, mps)

    #calculate normalization constant C
    C = torch.zeros([1,1], dtype=torch.cdouble)
    for i in range(2):
        prod = torch.matmul(mps[0][:,:,i], torch.transpose(torch.conj(mps[0][:,:,i]), 0,1))
        C = C + prod

    #normalize mps by adapting first mps element
    normConst = pow(C, 1./2)
    mps[0] = mps[0]*(1./normConst)

    #return normalization constant, e.g. for debugging
    return(np.abs(C).item())

#calculate energy expectation value
#expect real output since H is hermitian
def getEnergyExpVal(mps, hamiltonian):
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
                b = hamiltonian[i+1][:,:,getDecRep([j,k])]
                c = mps[i][:,:,k]
                if(j==0 and k==0):
                    T = kronProd(a, kronProd(b,c))
                else:
                    T = torch.add(T, kronProd(a, kronProd(b,c)))
        prod = torch.matmul(prod, T)

    #multiply product of transfer matrices prod with boundary vectors
    res = torch.matmul(hamiltonian[0], torch.matmul(prod, hamiltonian[N+1]))

    #check whether imaginary part is small enough (expect result to be real)
    if (res.imag.abs() > 1e-10):
        print("Warning: Imaginary part > 1e-10")

    return(res.real.numpy()[0,0])

#normalize and truncate bond dimensions of given mps
#truncation: only those singular values are kept that if
#contributing to the discarded weight do not surpass the
#given threshold eps and in number below Dmax

#returns the arithmetic mean of the truncated bond
#dimension of all MPS elements
def truncateMPS(mps, eps, Dmax=10):
    length = len(mps)

    #right-normalize MPS
    right_normalizeMPS(mps)

    mean_s = 0
    for i in range(length-1):
        #reshaping of tensor mps[i] into matrix A of dimensions (a_i,s_i) x a_i+1
        #and mps[i+1] into matrix C of dimension a_i+1 x (a_i+2, s_i+1)
        ai  = mps[i].size()[0]
        ai1 = mps[i].size()[1]
        ai2 = mps[i+1].size()[1]

        A = torch.zeros([2*ai, ai1], dtype=torch.cdouble)
        for j in range(ai):
            A[2*j,  :] = mps[i][j,:,0]
            A[2*j+1,:] = mps[i][j,:,1]

        C = torch.zeros([ai1, 2*ai2], dtype=torch.cdouble)
        for j in range(ai2):
            C[:,  2*j] = mps[i+1][:,j,0]
            C[:,2*j+1] = mps[i+1][:,j,1]

        #perform SVD on matrix A
        u, s, vh = np.linalg.svd(A.numpy(), full_matrices=False)

        #-------------------------------------------------------
        #truncation: only those singular values are kept that if
        #contributing to the discarded weight do not surpass the
        #given threshold eps
        # err = 0 #discarded weight squared
        # num_s = 0 #number of singular values after truncation
        # for j in range(len(s)):
        #     err = err + s[len(s)-1-j]**2
        #     num_s = len(s) - 1 - j
        #     if(err > eps**2):
        #         num_s = num_s + 1
        #         break
        #
        # if number of singular values exceeds a given maximal
        # bond dimension Dmax, truncate all singular values up
        # to Dmax to limit computational effort
        # if(num_s > Dmax):
        #     num_s = Dmax
        #
        # new_s = s[0:num_s]


        #truncate all singular values exceeding a given maximal
        #bond dimension mpar
        num_s = s.size
        if(num_s > Dmax):
            num_s = Dmax
            new_s = s[0:num_s]
        else:
            new_s = s
        #-------------------------------------------------------

        #if new_s is empty, take the first singular value
        if not np.any(new_s):
            new_s = np.array(s[0])
            num_s = 1

        #calculate mean bond dimension mean_s given by arithmetic mean
        #of truncated bond dimension of every MPS element
        mean_s += num_s

        #truncate other SVD matrices u, vh
        #u: truncate to num_s columns
        #vh: truncate to num_s rows
        idx = np.arange(0, num_s, 1)
        new_u = u[:, idx]
        new_vh = vh[idx, :]

        #recasting objects from numpy arrays to pytorch tensors
        ut = torch.from_numpy(new_u)
        st = torch.from_numpy(new_s)
        vht = torch.from_numpy(new_vh)

        #get matrix s*v*C using complex matrix multiplication of numpy
        if(num_s == 1):
            sv_tmp = new_s * new_vh
        else:
            sv_tmp = np.matmul(np.array(torch.diag(st).numpy()), vht.numpy())
        svc_n = np.matmul(sv_tmp, C.numpy())
        svc = torch.from_numpy(svc_n)

        #define new MPS tensor elements
        A_tilde = torch.zeros([ai, num_s, 2], dtype=torch.cdouble)
        nextA_tilde =torch.zeros([num_s, ai2, 2], dtype=torch.cdouble)

        #reshaping ut into new MPS tensors with the given prescription
        for j in range(ai):
            A_tilde[j,:,0] = ut[2*j,:]
            A_tilde[j,:,1] = ut[2*j+1,:]

        for j in range(ai2):
            nextA_tilde[:,j,0] = svc[:,2*j]
            nextA_tilde[:,j,1] = svc[:,2*j+1]

        #set new MPS elements
        mps[i] = A_tilde
        mps[i+1] = nextA_tilde

    mean_s = mean_s / (length-1)
    return mean_s

#calculate expectation value arbitrary MPO mpo for state mps
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
#-----------------------------------------------------------------------

'''exact diagonalization of Hamiltonian'''
#-----------------------------------------------------------------------
#get N-particle Hamiltonian as two matrix components H_0 and H_1
def get_H_comp(N, J, beta):
    dim = pow(2,N)
    h0 = torch.zeros([dim,dim], dtype=torch.cdouble)
    h1 = torch.zeros([dim,dim], dtype=torch.cdouble)

    #sigma_x
    sigmaX = torch.zeros([2,2], dtype=torch.cdouble)
    sigmaX[0,1] = 1
    sigmaX[1,0] = 1

    #sigma_z
    sigmaZ = torch.zeros([2,2], dtype=torch.cdouble)
    sigmaZ[0,0] = 1
    sigmaZ[1,1] = -1

    #identity
    id2 = torch.zeros([2,2], dtype=torch.cdouble)
    id2[0,0] = 1
    id2[1,1] = 1

    for i in range(N-1):
        for j in range(N):
            if (i == (N-2)):
                if (j==0):
                    lastterm = sigmaZ
                else:
                    lastterm = kronProd(id2, lastterm)
            if (j==0 and i != N-2):
                kprod0 = id2
                kprod1 = id2
                continue
            elif(j==0 and i == N-2):
                kprod0 = sigmaX
                kprod1 = id2
                continue
            if (j > (N-1)-i):
                kprod0 = kronProd(id2, kprod0)
            else:
                kprod0 = kronProd(sigmaX, kprod0)
            if (j == (N-1)-i):
                kprod1 = kronProd(sigmaZ, kprod1)
            else:
                kprod1 = kronProd(id2, kprod1)
        h0 = torch.add(h0, kprod0)
        h1 = torch.add(h1, kprod1)
    h1 = torch.add(h1,lastterm)

    return J*h0, beta*h1;

#exact diagonalization for N particle Hamiltonian H
#calculate eigenstates and corresponding eigenvalues of H
def getEigenstate(N, J, beta):
    H0, H1 = get_H_comp(N, J, beta)
    hamiltonian = H0 + H1
    del H0
    del H1

    eigenvals, eigenvecs = np.linalg.eig(hamiltonian.numpy())
    teigenvals = torch.from_numpy(eigenvals)
    teigenvecs = torch.from_numpy(eigenvecs)

    del hamiltonian
    gc.collect()
    return teigenvals, teigenvecs;

#print the eigenvalues and corresponding eigenstates in
#ascending order
def printEigenstate(N, J, beta, exactH=True, delta_t=0, ST=1):
    if(exactH):
        eigenvals, eigenvecs = getEigenstate(N, J, beta)
    if not (exactH):
        eigenvals, eigenvecs = get_trotterized_H(N, J, beta, delta_t, ST)
    evals = eigenvals.numpy()
    evecs = eigenvecs.numpy()
    #sort eigenvalues and corresponding eigenvectors in
    #ascending order
    iid = evals.argsort()[::]
    evals = evals[iid]
    evecs = evecs[:,iid]

    if(exactH):
        print("----------------Eigenstates----------------")
    if not(exactH):
        print("---------Approximated Eigenstates----------")
    for i in range(len(evals)):
        print("Eigenvalue: ", evals[i].real)
        print("to eigenstate: ", evecs[:,i].real, "\n")

#calculate ground state from exact numerical diagonalization
def gs_from_exact_diag(N, J, beta):
    H0, H1 = get_H_comp(N, J, beta)
    hamiltonian = H0 + H1
    del H0
    del H1
    evals, evecs = np.linalg.eig(hamiltonian.numpy())

    #sort eigenvalues and corresponding eigenvectors in ascending order
    iid = evals.argsort()
    evals = evals[iid]
    evecs = evecs[:,iid]

    gs = evecs[:,0]
    E0 = evals[0]

    del hamiltonian
    gc.collect()

    return E0, gs;
#------------------------------------------------------------------------

'''trotterized Hamiltonian'''
#------------------------------------------------------------------------
#calculate trotterized Hamiltonian for N particles
def get_trotterized_H(N, J, beta, delta_t, ST=1):
    #check whether to perform ST 1 or ST 2
    assert(ST == 1 or ST == 2), "Wrong input for ST!"

    #dimension of Hamiltonian
    dim = pow(2,N)

    #calculate components of Hamiltonian
    H0, H1 = get_H_comp(N, J, beta)
    #calculate exponents
    h0 = -1*H0*delta_t
    h1 = -1*H1*delta_t

    #calculate matrix exponential and
    #product of exponentials for ST 1 or ST 2
    if (ST == 1):
        U1 = linalg.expm(h0.numpy())
        U2 = linalg.expm(h1.numpy())
        U = np.matmul(U1,U2)
    elif (ST == 2):
        U1 = linalg.expm(h0.numpy())
        h1 = 0.5*h1
        U2 = linalg.expm(h1.numpy())
        U = np.matmul(U2, np.matmul(U1, U2))

    evals, evecs = np.linalg.eig(U)

    nevals = np.log(evals.real)
    #normalize eigenvalues by the factor of -1/delta_t
    nevals = -1/delta_t*nevals

    teigenvals = torch.from_numpy(nevals)
    teigenvecs = torch.from_numpy(evecs)

    return teigenvals, teigenvecs;
#-----------------------------------------------------------------------

'''=============================================================================
 time evolving block decimation (TEBD):
============================================================================='''
#generate 2-particle gate as tensor of rank 4
#if mode = 0: obtain 2-particle gates for iTEBD
#if mode = 1: obtain 2-particle gates for  real TEBD
def gen2particleGatesU(mode, J, beta, delta_t):

    if(mode==0):
        #define 2-particle time evolution operator elements for iTEBD
        cJ = cmath.cosh(J*delta_t)
        sJ = cmath.sinh(J*delta_t)
        eB = cmath.exp(beta*delta_t/2)
        meB = cmath.exp(-1*beta*delta_t/2)
    elif(mode==1):
        #define 2-particle time evolution operator elements for real TEBD
        cJ = cmath.cos(J*delta_t)
        sJ = complex(0.,1.)*cmath.sin(J*delta_t)
        eB = cmath.exp(complex(0.,1.)*beta*delta_t/2)
        meB = cmath.exp(-1*complex(0.,1.)*beta*delta_t/2)

    #define 2-particle time evolution operators as tensors
    #index order: til(s)_i, til(s)_i+1, s_i, s_i+1:
    #nearest neighbour interactions (pairwise)
    U1 = torch.zeros([2,2,2,2], dtype=torch.cdouble)
    #magnetic field terms (pairwise)
    U2 = torch.zeros([2,2,2,2], dtype=torch.cdouble)
    #magnetic field contribution for last particle at position N-1
    U3 = torch.zeros([2,2,2,2], dtype=torch.cdouble)

    #U1[til(s)_i, til(s)_i+1, s_i, s_i+1]
    U1[0,0,0,0] =    cJ
    U1[0,1,0,1] =    cJ
    U1[1,0,1,0] =    cJ
    U1[1,1,1,1] =    cJ

    U1[0,0,1,1] = -1*sJ
    U1[0,1,1,0] = -1*sJ
    U1[1,0,0,1] = -1*sJ
    U1[1,1,0,0] = -1*sJ

    #U3[til(s)_i, til(s)_i+1, s_i, s_i+1]
    U2[0,0,0,0] =    meB
    U2[0,1,0,1] =    meB
    U2[1,0,1,0] =    eB
    U2[1,1,1,1] =    eB

    #U4[til(s)_i, til(s)_i+1, s_i, s_i+1]
    U3[0,0,0,0] =    meB
    U3[0,1,0,1] =    eB
    U3[1,0,1,0] =    meB
    U3[1,1,1,1] =    eB

    return U1, U2, U3;

'''-----------------------------------------------------------------------------
imaginary-time evolution algorithm for computing the ground state MPS
-----------------------------------------------------------------------------'''
#contraction of MPS mps at position pos and pos+1 with 2-particle gate U
def evolve2state(U, delta_t, mps, pos):

    assert(pos < (len(mps)-1)), "Specified particle is greater than particle number!"

    #mps elements have index structure: a_i, a_i+1, s_i = ijk (example)
    #U has index structure: til(s)_i, til(s)_i+1, s_i, s_i+1 = opkm
    #delta has index structure: til(s)_i, a_i, til(s)_i+1, a_i+2 = oipl
    delta = torch.einsum('ijk,jlm,opkm->oipl', mps[pos], mps[pos+1], U)
    #max(a_i)+1:
    ai = (delta.size()[1])
    #max(a_i+2)+1:
    aii = (delta.size()[3])

    #reshape delta into matrix delta1 with indices (til(s)_i, a_i),(til(s)_i+1, a_i+2)
    #using the prescription given above
    delta1 = torch.zeros([2*ai, 2*aii], dtype=torch.cdouble)
    for i in range(2*ai):
        for j in range(2*aii):
            if(i < ai and j < aii):
                delta1[i,j] = delta[0,i,0,j]
            elif (i < ai):
                delta1[i,j] = delta[0,i,1,j-aii]
            elif (j < aii):
                delta1[i,j] = delta[1,i-ai,0,j]
            else:
                delta1[i,j] = delta[1,i-ai,1,j-aii]

    #perform SVD on matrix delta1 using numpy for complex matrices
    u, s, vh = np.linalg.svd(delta1.numpy(), full_matrices=False)
    #print(u.shape, s.shape, vh.shape)
    #print(np.allclose(delta1.numpy(), np.dot(u * s, vh)))

    #recasting objects from numpy arrays to pytorch tensors
    ut = torch.from_numpy(u)
    st = torch.from_numpy(s)
    vht = torch.from_numpy(vh)

    #get matrix s*v using complex matrix multiplication of numpy
    sv_tmp = np.matmul(torch.diag(st).numpy(), vht.numpy())
    sv = torch.from_numpy(sv_tmp)

    #calculate s*v
    #s_diag = torch.add(torch.diag(st), torch.zeros([st.size()[0], st.size()[0]], dtype=torch.cdouble))
    #sv1 = torch.matmul(s_diag, vht)

    #number of singular values
    num_s = st.size()[0]

    #define new MPS tensor elements
    A_tilde = torch.zeros([ai, num_s, 2], dtype=torch.cdouble)
    nextA_tilde =torch.zeros([num_s, aii, 2], dtype=torch.cdouble)

    #reshape matrices u and sv to new MPS tensors with the given prescription
    for i in range(2*ai):
        if(i < ai):
            A_tilde[i,:,0] = ut[i,:]
        else:
            A_tilde[i-ai,:,1] = ut[i,:]

    for j in range(2*aii):
        if(j < aii):
            nextA_tilde[:,j,0] = sv[:,j]
        else:
            nextA_tilde[:,j-aii,1] = sv[:,j]

    #set new MPS elements
    mps[pos] = A_tilde
    mps[pos+1] = nextA_tilde

    return(mps)

#calculate ground state applying the corresponding 2-particle gates sequentially
#determine ground state MPS with evaluation purpose
def getGroundstateMPSEval(bonddims, J, beta, dt, Dmax, printinfo=True, eps= 1e-6):
    #particle number
    N = len(bonddims)

    #random normalized MPS to start iteration
    start = genMPS(bonddims)
    right_normalizeMPS(start)

    #hamiltonian
    hamiltonian = genH(len(start), J, beta)

    #generate 2-particle gates U1, U2, U3
    U1, U2, U3 = gen2particleGatesU(0, J, beta, dt)

    #store energy expectation values in a list
    energies = []
    energy = getEnergyExpVal(start, hamiltonian)
    energies.append(energy)

    #relative errors energy per step
    rel_err = 1

    #diagonalize H trotterized with order ST:
    eigenvals, eigenvecs = get_trotterized_H(N, J, beta, dt, ST=2)
    evals = eigenvals.numpy()
    evecs = eigenvecs.numpy()

    #sort eigenvalues and corresponding eigenvectors in
    #ascending order
    iid = evals.argsort()[::]
    evals = evals[iid]
    evecs = evecs[:,iid]
    tevecs = torch.from_numpy(evecs)

    #apply 2-particle time evolution gates sequentially
    tmp = start;

    #absolute time t
    tvals = [0]

    #save amplitudes of eigenstates
    res_amp = np.zeros((2, pow(2,N)), dtype=np.cdouble)

    i = 0
    while(np.abs(rel_err) > eps):
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
        truncateMPS(tmp, 1e-15, Dmax)

        energy = getEnergyExpVal(tmp, hamiltonian)
        energies.append(energy)

        #calculate absolute time t from time steps
        tvals.append(tvals[-1] + dt)

        #calculate relative error energy per step
        if (i != 0):
            rel_err = (energies[i] - energies[i-1])/energies[i-1]

        #calculate coeffcients of state for evaluation purpose
        coeff = calc_coeff_state(tmp)
        amp = np.array([])
        if(i < 2):
            for j in range(pow(2,N)):
                res_amp[i, j] = np.matmul(np.conj(evecs[:,j]), coeff)
        else:
            for j in range(pow(2,N)):
                amp = np.append(amp,np.matmul(np.conj(evecs[:,j]), coeff))
            res_amp = np.vstack((res_amp, amp))

        #increase iteration step
        i = i + 1

    steps = i
    if(printinfo):
        print("\nIteration steps: ", steps)
        print("\nEnergy ground state: ", energy)

    #store energy expectation values for every step in a file
    fl = open('energy_vs_it_step.txt', 'w')
    for i in range(len(energies)):
        fl.write("%4.3f %30.25f\n" % (tvals[i], np.abs((energies[i]-evals[0])/evals[0]).item()))
    fl.close()

    #store absolute value amplitudes for every step in a file
    for i in range(pow(2,N)):
        fl1 = open('amplitude' + str(i) + '.txt', 'w')
        for j in range(steps):
            fl1.write("%4.3f %30.25f\n" % (tvals[j], np.abs(res_amp[j,i]).item()))
        fl1.close()
    return(tmp)

#-------------------------------------------------------------------------------
#parameters: given by config-file conf from dataclass ImagiTime_config
#conf = ImagiTime_config(N=N, J=J, beta=beta, dt0=dt0, eps_trunc=eps_trunc, Dmax=Dmax, eps_dt=eps_dt, dt_red=dt_red, eps_tol=eps_tol, phys_gap=phys_gap, bdim_start=bdim_start)
@dataclass
class ImagiTime_config:
    N:          int          #particle number
    J:          float        #nearest neighbour coupling strength
    beta:       float        #external field coupling strength
    eps_trunc:  float = 1e-6 #truncation threshold discarded weights
    Dmax:       int   = 10   #maximal bond dimension after truncation
    dt0:        float = 0.4  #step size at t=0 (start)
    eps_dt:     float = 0.09 #convergence threshold for steps size dt
    dt_red:     float = 0.8 #value by which the step size is reduced
    eps_tol:    float = 0.001 #tolerance threshold of exponential error
    phys_gap:   float = 0.001  #gap between ground state & 1st excited level
    bdim_start: int   = 2    #bond dimension at t=0 (start)

#goal: calculate ground state mps, ground state energy

#procedure: use TEBD with additional
# -> dt-reduction (decrease number of iteration steps, decrease Trotter error)
# -> corresponding convergence criterion (dt smaller than precision eps_dt)
# -> direct truncation via SVD according to the discarded weight procedure

#output:
# -> E0: from Trotter error extrapolated energy (calculated at every reduction step)
# -> E: computed ground state energy MPS
# -> ground state MPS

#additional: create file "./out/dump/run_groundstate_N{N}_J{J}_beta{beta}_{now}.txt"
# 1.column:
# code word for reading file: [conf], [info], [update_dt], [update_E0], [itstep], [output]

# other columns:
# [1] = iteration step i, [2] = mean bond dimension MBD, [3] = current step size dt
# [4] = current energy E, [5] = relative error energy rel_err
# [6] = exponential error exp_err, [7] = slope energy slope

#obtain absolute time t by taking cumsum of dt values
def getGroundstate(conf, date_stamp):
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
    dump_stream = open(f"./out/run_groundstate_N{conf.N}_J{conf.J}_beta{conf.beta}_{date_stamp}.txt", 'w')
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

    print(f"Iteration steps: {i}")
    print(f"Energy ground state: {energy:15.10f}")
    print(f"Extrapolated ground state energy: {E0: f}\n")

    dump_stream.close()
    return E0, energy, tmp;
#-------------------------------------------------------------------------------

'''extract data'''
#-------------------------------------------------------------------------------
#search for string_to_search in from_file, extract columns at pos into to_file
#return columns x at pos[0] in from_file, and y at pos[1] in from_file
def extract_params_from_file(from_file, to_file, string_to_search, pos):
    x = []
    y = []
    #open the file in read only mode
    with open(from_file, 'r') as read_obj:
        for line in read_obj:
            if string_to_search in line:
                line_content = line.rstrip()
                x.append(float(line_content.split()[pos[0]]))
                y.append(float(line_content.split()[pos[1]]))

    #store params in file
    fl = open(to_file, 'w')
    for i in range(len(x)):
        fl.write(f"{x[i]} {y[i]}\n")
    fl.close()

    return x, y;
#-----------------------------------------------------------------------

'''-----------------------------------------------------------------------------
Real-time Evolution
-----------------------------------------------------------------------------'''
#additional: create file "./out/dump/run_time_evol_eval_N{N}_J{J}_beta{beta}_t{evol_time}_{now}.txt"
# 1.column:
# code word for reading file: [conf], [info], [itstep]

# other columns:
# [1] = iteration step i, [2] = mean bond dimension MBD,
# [3] = current energy (only for evolveInTimeEval)

def evolveInTime(start_state, J, beta, dt, steps, eps_trunc=1e-4, Dmax=10):
    #duration time evolution
    evol_time = steps * dt

    #particle number
    N = len(start_state)

    #time evolution operators as 2-particle gates
    U1, U2, U3 = gen2particleGatesU(1, J, beta, dt)

    #current date stamp
    now = datetime.now().strftime("%y%m%d%H%M%S")

    #Dumping all the output into file
    dump_stream = open(f"./out/run_time_evol_N{N}_J{J}_beta{beta}_t{evol_time}_{now}.txt", 'w')
    dump_stream.write(f"{'[conf]':12s} {'N':>10s} {N:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'J':>10s} {J:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'beta':>10s} {beta:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'dt':>10s} {dt:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'steps':>10s} {steps:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'eps_trunc':>10s} {eps_trunc:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'Dmax':>10s} {Dmax:>15d}\n")

    #dump_stream.write(f"{'[info]':12s} {'i':>5s} {'MBD':>8s} {'E':>17s}\n")
    dump_stream.write(f"{'[info]':12s} {'step':>5s} {'MBD':>8s}\n")

    #apply 2-particle time evolution gates sequentially
    tmp = start_state;

    for i in range(steps):
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U1, dt, tmp, j)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)

        #normalization & truncation
        mean_bonddim = truncateMPS(tmp, eps_trunc, Dmax)

        #dump_stream.write(f"{'[itstep]':12s} {i:5d} {mean_bonddim:8.3f} {energy:17.10f}\n")
        dump_stream.write(f"{'[itstep]':12s} {i:5d} {mean_bonddim:8.3f}\n")

    return tmp;

#calculate energy expectation value current mps for every time step
def evolveInTimeEval(start_state, J, beta, dt, steps, eps_trunc=1e-4, Dmax=10):
    #duration time evolution
    evol_time = steps * dt

    #particle number
    N = len(start_state)

    #hamiltonian
    hamiltonian = genH(len(start_state), J, beta)

    #time evolution operators as 2-particle gates
    U1, U2, U3 = gen2particleGatesU(1, J, beta, dt)

    #store energy expectation values in a list
    energy = getEnergyExpVal(start_state, hamiltonian)

    #current date stamp
    now = datetime.now().strftime("%y%m%d%H%M%S")

    #Dumping all the output into file
    dump_stream = open(f"./out/run_time_evol_eval_N{N}_J{J}_beta{beta}_t{evol_time}_dt{dt}_{now}.txt", 'w')
    dump_stream.write(f"{'[conf]':12s} {'N':>10s} {N:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'J':>10s} {J:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'beta':>10s} {beta:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'dt':>10s} {dt:>15.3f}\n")
    dump_stream.write(f"{'[conf]':12s} {'steps':>10s} {steps:>15d}\n")
    dump_stream.write(f"{'[conf]':12s} {'eps_trunc':>10s} {eps_trunc:>15.3e}\n")
    dump_stream.write(f"{'[conf]':12s} {'Dmax':>10s} {Dmax:>15d}\n")

    dump_stream.write(f"{'[info]':12s} {'i':>5s} {'MBD':>8s} {'E':>17s}\n")

    #apply 2-particle time evolution gates sequentially
    tmp = start_state;

    for i in range(steps):
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U1, dt, tmp, j)
        for j in range(len(tmp)-1):
            tmp = evolve2state(U2, dt, tmp, j)
        tmp = evolve2state(U3, dt, tmp, len(tmp)-2)

        #normalization & truncation
        mean_bonddim = truncateMPS(tmp, eps_trunc, Dmax)

        energy = getEnergyExpVal(tmp, hamiltonian)

        dump_stream.write(f"{'[itstep]':12s} {i:5d} {mean_bonddim:8.3f} {energy:17.10f}\n")

    return energy, tmp;
#-----------------------------------------------------------------------

'''=============================================================================
Observables
============================================================================='''

'''Magnetization'''
#-----------------------------------------------------------------------
#calculate explicit matrix form of magnetization operator for N particles
#for longitudinal antimagnetization
def calcAntiMagnetMPO(N):
    dim = pow(2,N)
    M = torch.zeros([dim,dim], dtype=torch.cdouble)

    #sigma_x matrix definition
    sigmaX = torch.zeros([2,2], dtype=torch.cdouble)
    sigmaX[0,1] = 1
    sigmaX[1,0] = 1

    #identity 2x2 matrix definition
    id2 = torch.zeros([2,2], dtype=torch.cdouble)
    id2[0,0] = 1
    id2[1,1] = 1

    #first matrix terms
    kprod1 = kronProd(sigmaX, id2)
    kprod2 = kronProd(id2, sigmaX)

    if(N==1):
        M = torch.add(M, sigmaX)

    #calculate explicit matrix form using factorisation into particles subspaces
    for i in range(N):
        for j in range(N-1):
            if(j==0 and i==0):
                prod = kprod1
            elif(i==N-1 and j==N-2):
                if(N==2):
                    prod = kprod2
                else:
                    prod = kronProd(prod, kprod2)
            elif(j==0):
                prod = id2
            elif(j==i):
                prod = kronProd(prod, kprod1)
            else:
                prod = kronProd(prod, id2)
        #odd lattice sites pick up a minus sign
        if(i%2):
            prod = (-1)*prod
            M = torch.add(M, prod)
        else:
            M = torch.add(M, prod)

    return M/N;

#calculate explicit matrix form of magnetization operator for N particles
#for transversal magnetization
def calcMagnetMPO(N):
    dim = pow(2,N)
    M = torch.zeros([dim,dim], dtype=torch.cdouble)

    #sigma_z matrix definition
    sigmaZ = torch.zeros([2,2], dtype=torch.cdouble)
    sigmaZ[0,0] = 1
    sigmaZ[1,1] = -1

    #identity 2x2 matrix definition
    id2 = torch.zeros([2,2], dtype=torch.cdouble)
    id2[0,0] = 1
    id2[1,1] = 1

    #first matrix terms
    kprod1 = kronProd(sigmaZ, id2)
    kprod2 = kronProd(id2, sigmaZ)

    if(N==1):
        M = torch.add(M, sigmaZ)

    #calculate explicit matrix form using factorisation into particles subspaces
    for i in range(N):
        for j in range(N-1):
            if(j==0 and i==0):
                prod = kprod1
            elif(i==N-1 and j==N-2):
                if(N==2):
                    prod = kprod2
                else:
                    prod = kronProd(prod, kprod2)
            elif(j==0):
                prod = id2
            elif(j == i):
                prod = kronProd(prod, kprod1)
            else:
                prod = kronProd(prod, id2)
        M = torch.add(M, prod)

    return (-1)*M/N;

#determine expectation value magnetization operator for a state given in local basis
# if long=True: calculate longitudinal antimagnetization
# if long=False: calculate transversal magnetization
def getMagnetExpValFromState(N, state, long):
    if(long):
        M = calcAntiMagnetMPO(N)
    else:
        M = calcMagnetMPO(N)
    magnetization = np.matmul(state.conjugate().transpose(), np.matmul(M, state)).numpy().real
    return magnetization;

#determine magnetization operator as MPO for N particles
# if long=True: calculate longitudinal antimagnetization
# if long=False: calculate transversal magnetization
def getMagnet(N, long):
    #define tensor B for different magnetizations
    B = torch.zeros([3,3, 4], dtype=torch.cdouble)
    if(long):
        #B00:
        B[0,0,0] = 1
        B[2,2,0] = 1
        #B01:
        B[0,2,1] = -1
        #B10:
        B[0,2,2] = -1
        #B11:
        B[0,0,3] = 1
        B[2,2,3] = 1
    else:
        #B00:
        B[0,0,0] = 1
        B[2,2,0] = 1
        B[0,2,0] = 1
        #B11:
        B[0,0,3] = 1
        B[2,2,3] = 1
        B[0,2,3] = -1

    #define left boundary vector
    Brows = B.size()[0]
    Bcols = B.size()[1]
    vL = torch.zeros([1,Brows], dtype=torch.cdouble)
    vL[0,0] = 1.

    #define right boundary vector
    vR = torch.zeros([Bcols,1], dtype=torch.cdouble)
    vR[(Bcols-1),0] = 1.

    mpo = []
    mpo.append(vL)
    for k in range(N):
        #take sign (-1)^k into account for each lattice site k
        #for the longitudinal antimagnetization
        if(long):
            B[0,2, 1] = complex(-1,0)*B[0,2, 1]
            B[0,2, 2] = complex(-1,0)*B[0,2, 2]
        mpo.append(B.detach().clone())
    mpo.append(vR)

    return mpo
#-----------------------------------------------------------------------

'''One-point correlation function'''
#-----------------------------------------------------------------------
#determine one-point correlation operator as MPO for N particles
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
#-----------------------------------------------------------------------

'''=============================================================================
Entanglement Entropy
============================================================================='''

#-----------------------------------------------------------------------
#calculate entanglement entropy for different bond dimensions, different length L of subsystem
def calc_entanglement_entropy(L, mps):
    central_gauge(L, mps)

    #reshape tensor at position L+1 to matrix of dimension a_L+1 x (s_L+1, a_L+2):
    ai = mps[L+1].size()[0]
    ai1 = mps[L+1].size()[1]

    A = torch.zeros([ai, 2*ai1], dtype=torch.cdouble)
    for j in range(2*ai1):
        if(j < ai1):
            A[:,j] = mps[L+1][:,j, 0]
        else:
            A[:,j] = mps[L+1][:,j-ai1, 1]

    #perform SVD on site L+1
    u, s, vh = np.linalg.svd(A.numpy(), full_matrices=False)

    s2 = s*s
    l = np.log2(s2)

    S = 0
    for i in range(len(s2)):
        prod = s2[i]*l[i]
        S = S + prod

    return(-1*S)
#-----------------------------------------------------------------------

'''=============================================================================
Singular Values
============================================================================='''
#determine singular values of mps at position pos
def get_sing_vals(pos, mps):
    #right-normalize MPS
    right_normalizeMPS(mps)

    for i in range(pos):
        left_gauge(i, mps)

    #reshaping of tensor mps[pos] into matrix A of dimensions (a_i,s_i) x a_i+1
    ai  = mps[pos].size()[0]
    ai1 = mps[pos].size()[1]

    A = torch.zeros([2*ai, ai1], dtype=torch.cdouble)
    for j in range(ai):
        A[2*j,  :] = mps[pos][j,:,0]
        A[2*j+1,:] = mps[pos][j,:,1]

    #perform SVD on matrix A
    u, s, vh = np.linalg.svd(A.numpy(), full_matrices=False)

    return s
