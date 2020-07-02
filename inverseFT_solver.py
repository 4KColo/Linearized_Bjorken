#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
from functools import partial
import ctypes
from time import time

C1 = 0.197327   # 1 GeV*fm = 0.197327
viscosity = 0	# eta/s in units of 1/4pi

### read in file
filename = 'SolverTest_Gauss_Nx=251_Neta=251.hdf5'
f = h5py.File(filename, 'r')
    
groupname = '4pi_eta_over_s='+str(viscosity)
if groupname in f:
    group = f[groupname]
    kx = group.attrs['kx']
    ky = group.attrs['ky']
    keta = group.attrs['keta']
    tau = group['tau'][()]
    tau = tau

Nstep = 5

d_kx = kx[1]-kx[0]
d_ky = ky[1]-ky[0]
d_keta = keta[1]-keta[0]
N_kx_old = len(kx)
N_ky_old = len(ky)
N_keta_old = len(keta)
N_add_xy = 180
N_add_eta = 180
kx_half = np.max(kx) + d_kx * N_add_xy
ky_half = np.max(ky) + d_ky * N_add_xy
keta_half = np.max(keta) + d_keta * N_add_eta
kx_max = kx_half*2.0
ky_max = ky_half*2.0
keta_max = keta_half*2.0
N_kx = N_kx_old + N_add_xy*2
N_ky = N_ky_old + N_add_xy*2
N_keta = N_keta_old + N_add_eta*2
kx_new = np.linspace(-kx_half, kx_half, N_kx)
ky_new = np.linspace(-ky_half, ky_half, N_ky)
keta_new = np.linspace(-keta_half, keta_half, N_keta)
factor = kx_max*ky_max*keta_max/(2.*np.pi)**3

Nx_half = int(N_kx/2)
Ny_half = int(N_ky/2)
Neta_half = int(N_keta/2)
xlist = np.array(range(-Nx_half, Nx_half+1))*2.0*np.pi/kx_max
ylist = np.array(range(-Ny_half, Ny_half+1))*2.0*np.pi/ky_max
etalist = np.array(range(-Neta_half, Neta_half+1))*2.0*np.pi/keta_max


### save file
filename2 = 'Inverse_FT_Gauss.hdf5'
if not os.path.exists(filename2):
    f2 = h5py.File(filename2, 'a')
else:
    f2 = h5py.File(filename2, 'a')
    
groupname2 = '4pi_eta_over_s='+str(int(viscosity))

if groupname2 in f2:
    del f2[groupname2]
group2 = f2.create_group(groupname2)
group2.attrs.create('x', xlist)
group2.attrs.create('y', ylist)
group2.attrs.create('eta', etalist)
group2.create_dataset('tau', data = tau)


for istep in range(Nstep):
    print(istep)
    
    t0 = time()
    
    data = group['e_and_g'][()]
    e_tilde = data[:,:,:,0,istep]
    geta_tilde = data[:,:,:,1,istep]*1j
    
    t1 = time()
    print(t1 - t0)
    
    # extend to larger momentum space
    e_tilde = np.pad(e_tilde, ((N_add_xy,N_add_xy),(N_add_xy,N_add_xy),(N_add_eta,N_add_eta)), mode='constant', constant_values = 0)
    geta_tilde = np.pad(geta_tilde, ((N_add_xy,N_add_xy),(N_add_xy,N_add_xy),(N_add_eta,N_add_eta)), mode='constant', constant_values = 0)
    
    t2 = time()
    print(t2 - t1)
    
    e_tilde = np.fft.ifftshift(e_tilde)
    e = np.fft.ifftn(e_tilde) * factor
    e = np.fft.fftshift(e)
     
    geta_tilde = np.fft.ifftshift(geta_tilde)
    geta = np.fft.ifftn(geta_tilde) * factor
    geta = np.fft.fftshift(geta)
    
    t3 = time()
    print(t3 - t2)
    
    group2.create_dataset('e'+str(istep), data = e)
    group2.create_dataset('geta'+str(istep), data = geta)
    
f2.close()

f.close()
