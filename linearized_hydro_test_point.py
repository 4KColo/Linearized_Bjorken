#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
from functools import partial
import ctypes
from time import time
from evolution import Tau_to_Temp
from evolution import RK4
from evolution import Point_Source_Test
from constants import *

### parameters
x0 = 0.0    # GeV^-1
y0 = 0.0
eta0 = 0.0
delta_E = 50.0      	# Delta E, unit: GeV
viscosity = 1.0			# eta/s in units of 1/4pi
viscosity_over_s = viscosity/(4.0*np.pi)

dtau = 0.001/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)
list_temp = Tau_to_Temp(list_tau)

### parallel computing
Nx = 200
Ny = 200
Neta = 100
kx = np.linspace(-5.0, 5.0, Nx)        # GeV
ky = np.linspace(-5.0, 5.0, Ny)        # GeV
keta = np.linspace(-5.0, 5.0, Neta)

shared_array_base = mp.Array(ctypes.c_double, Nx*Ny*Neta*8)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(Nx,Ny,Neta,8)

def Linearized_Hydro(X, I_tau, List = shared_array):
    I_kx = X[0]
    I_ky = X[1]
    I_keta = X[2]
    Kx = kx[I_kx]
    Ky = ky[I_ky]
    Keta = keta[I_keta]
    if I_tau == 0:
        re_e_new, im_e_new = Point_Source_Test(Kx, Ky, Keta, x0, y0, eta0, delta_E, list_tau[I_tau])
        re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        re_e_old, im_e_old, re_gx_old, im_gx_old, re_gy_old, im_gy_old, re_geta_old, im_geta_old = List[I_kx][I_ky][I_keta]
        re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = RK4(re_e_old, im_e_old, re_gx_old, im_gx_old, re_gy_old, im_gy_old, re_geta_old, im_geta_old, Kx, Ky, Keta, list_tau[I_tau], dtau, cs_sqd, viscosity_over_s)
    
    List[I_kx][I_ky][I_keta] = np.array([re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new])
    return None



### save file
filename = 'Test_Point_Source_Nx='+str(Nx)+'_Neta='+str(Neta)+'.hdf5'
if not os.path.exists(filename):
    f = h5py.File(filename, 'a')
else:
    f = h5py.File(filename, 'a')
    
groupname = '4pi_eta_over_s='+str(int(viscosity))
if groupname in f:
    del f[groupname]
group = f.create_group(groupname)

group.attrs.create('x0', x0)
group.attrs.create('y0', y0)
group.attrs.create('eta0', eta0)
group.attrs.create('kx', kx)
group.attrs.create('ky', ky)
group.attrs.create('keta', keta)
group.create_dataset('tau', data = list_tau)

#ti = time()

Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
for istep in range(nstep):
    Update_Parallel = partial(Linearized_Hydro, I_tau = istep)
    pool.map(Update_Parallel, [(i,j,k) for i in range(Nx) for j in range(Ny) for k in range(Neta)])
    if istep%100 == 0:
        print istep
        group.create_dataset('e_g_'+str(istep), data = shared_array)
    

#tf = time()
#print tf - ti

f.close()
