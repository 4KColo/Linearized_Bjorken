#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
import ctypes
from time import time
from evolution import Tau_to_Temp
from evolution import RK4
from evolution import Point_Source_Test
from constants import *


x0 = 0.0    # GeV^-1
y0 = 0.0
eta0 = 0.0
delta_E = 10.0/tau_i	# Delta E/tau, units: GeV^2
viscosity = 0.0			# eta/s in units of 1/4pi
viscosity_over_s = viscosity/(4.0*np.pi)

dtau = 0.001/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)
list_temp = Tau_to_Temp(list_tau)


def Linearized_Hydro(Kx, Ky, Keta):
    re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = 0., 0., 0., 0., 0., 0., 0., 0.
    
    for i in range(nstep):
        # source for perturbations in the first step
        if i == 0:
        	re_e_source, im_e_source = Point_Source_Test(Kx, Ky, Keta, x0, y0, eta0, delta_E)
        	re_e_old  = re_e_new  + re_e_source
        	im_e_old  = im_e_new  + im_e_source
        else:
        	re_e_old  = re_e_new
        	im_e_old  = im_e_new
        re_gx_old = re_gx_new
        im_gx_old = im_gx_new
        re_gy_old = re_gy_new
        im_gy_old = im_gy_new
        re_geta_old = re_geta_new
        im_geta_old = im_geta_new
                        
        re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = RK4(re_e_old, im_e_old, re_gx_old, im_gx_old, re_gy_old, im_gy_old, re_geta_old, im_geta_old, Kx, Ky, Keta, list_tau[i], dtau, cs_sqd, viscosity_over_s)
        
    return np.array([re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new])
    

### parallel computing
Nx = 2
Ny = 2
Neta = 2
kx = np.linspace(-10.0, 10.0, Nx)	# GeV
ky = np.linspace(-10.0, 10.0, Ny)	# GeV
keta = np.linspace(-10.0, 10.0, Neta)

shared_array_base = mp.Array(ctypes.c_double, Nx*Ny*Neta*8)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(Nx,Ny,Neta,8)

def Parallel_LinearHydro(X, Output = shared_array):
    i = X[0]
    j = X[1]
    k = X[2]
    Output[i][j][k] = Linearized_Hydro(kx[i], ky[j], keta[k])
    return None
    

ti = time()

Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
pool.map(Parallel_LinearHydro, [(i,j,k) for i in range(Nx) for j in range(Ny) for k in range(Neta)])

tf = time()
print tf - ti


### save file
filename = 'Linearized_Hydro_over_Bjorken.hdf5'
if not os.path.exists(filename):
    f = h5py.File(filename, 'w')
else:
    f = h5py.File(filename, 'a')
    
groupname = '4pi_eta_over_s='+str(int(viscosity))
if groupname in f:
    del f[groupname]
group = f.create_group(groupname)

group.attrs.create('Nx', Nx)
group.attrs.create('Ny', Ny)
group.attrs.create('Neta', Neta)
group.attrs.create('kx', kx)
group.attrs.create('ky', ky)
group.attrs.create('keta', keta)
group.create_dataset('e_g', data = shared_array)

f.close()

