#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
import ctypes
from time import time
from evolution import Tau_to_Temp
from evolution import RK4
from evolution import Gauss_Source
from evolution import Point_Source
from constants import *

e_jet = 100.0               # initial jet energy in GeV
x0 = 0.0                    # initial position of the jet
y0 = 0.0
eta0 = 0.0

para_gauss_width = 1.       # sigma = #/(pi*T), width in Gauss energy deposition
kappa_sc = 1.               # parameter in the Eloss formula
viscosity_over_s = 0.1      # eta/s

tau_stop = 6.0/C1           # time of jet scaping QGP
dtau = 0.01/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
nstep_stop = int((tau_stop-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)

list_x = x0 + list_tau      # positions of the jet at each time step
list_y = y0 + 0.0*list_tau
list_eta = eta0 + 0.0*list_tau
list_temp = Tau_to_Temp(list_tau)


def Linearized_Hydro(Kx, Ky, Keta):
    re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = 0., 0., 0., 0., 0., 0., 0., 0.
    
    for i in range(nstep):
        # source for perturbations in each step
        re_e_source, im_e_source = Gauss_Source(Kx, Ky, Keta, list_tau[i], dtau, list_x[i], list_y[i], list_eta[i], e_jet, list_temp[i], kappa_sc, para_gauss_width)
        re_gx_source, im_gx_source = Gauss_Source(Kx, Ky, Keta, list_tau[i], dtau, list_x[i], list_y[i], list_eta[i], e_jet, list_temp[i], kappa_sc, para_gauss_width)
        re_e_old  = re_e_new  + re_e_source
        im_e_old  = im_e_new  + im_e_source
        re_gx_old = re_gx_new + re_gx_source
        im_gx_old = im_gx_new + im_gx_source
        re_gy_old = re_gy_new
        im_gy_old = im_gy_new
        re_geta_old = re_geta_new
        im_geta_old = im_geta_new
                        
        re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = RK4(re_e_old, im_e_old, re_gx_old, im_gx_old, re_gy_old, im_gy_old, re_geta_old, im_geta_old, Kx, Ky, Keta, list_tau[i], dtau, cs_sqd, viscosity_over_s)
        
    return np.array([re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new])
    

### parallel computing
N = 5
kx = np.linspace(-5.0, 5.0, N)
ky = np.linspace(-5.0, 5.0, N)
keta = np.linspace(-5.0, 5.0, N)

shared_array_base = mp.Array(ctypes.c_double, N*N*N*8)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(N,N,N,8)

def Parallel_LinearHydro(X, Output = shared_array):
    i = X[0]
    j = X[1]
    k = X[2]
    Output[i][j][k] = Linearized_Hydro(kx[i], ky[j], keta[k])
    return None
    

ti = time()

Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
pool.map(Parallel_LinearHydro, [(i,j,k) for i in range(N) for j in range(N) for k in range(N)])

tf = time()
print tf - ti


### save file
filename = 'Linearized_Hydro_over_Bjorken.hdf5'
if not os.path.exists(filename):
    f = h5py.File(filename, 'w')
else:
    f = h5py.File(filename, 'a')
    
groupname = 'eta_over_s='+str(viscosity_over_s)
if groupname in f:
    del f[groupname]
group = f.create_group(groupname)

group.attrs.create('N', N)
group.attrs.create('kx', kx)
group.attrs.create('ky', ky)
group.attrs.create('keta', keta)
group.create_dataset('e_g', data = shared_array)

f.close()

