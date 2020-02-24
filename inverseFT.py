#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
from functools import partial
import ctypes
from time import time

C1 = 0.197327   # 1 GeV*fm = 0.197327
viscosity = 1	# eta/s in units of 1/4pi

### read in file
filename = 'Test_Gauss_Source_Nx=225_Neta=225.hdf5'
f = h5py.File(filename, 'r')
    
groupname = '4pi_eta_over_s='+str(viscosity)
if groupname in f:
    group = f[groupname]
    kx = group.attrs['kx']
    ky = group.attrs['ky']
    keta = group.attrs['keta']
    tau = group['tau'][()]
    tau = tau*C1

Nstep = int(len(tau)/100.)+1

d_kx = kx[1]-kx[0]
d_ky = ky[1]-ky[0]
d_keta = keta[1]-keta[0]

N_kx_old = len(kx)
N_ky_old = len(ky)
N_keta_old = len(keta)
N_add = 250
kx_half = np.max(kx) + d_kx * N_add
ky_half = np.max(ky) + d_ky * N_add
keta_half = np.max(keta) + d_keta * N_add
kx_max = kx_half*2.0
ky_max = ky_half*2.0
keta_max = keta_half*2.0
N_kx = N_kx_old + N_add*2
N_ky = N_ky_old + N_add*2
N_keta = N_keta_old + N_add*2
kx_new = np.linspace(-kx_half, kx_half, N_kx)
ky_new = np.linspace(-ky_half, ky_half, N_ky)
keta_new = np.linspace(-keta_half, keta_half, N_keta)
factor = kx_max*ky_max*keta_max/(2.*np.pi)**3

Nx_half = N_kx/2
Ny_half = N_ky/2
Neta_half = N_keta/2
xlist = np.array(range(-Nx_half, Nx_half+1))*2.0*np.pi/kx_max
ylist = np.array(range(-Ny_half, Ny_half+1))*2.0*np.pi/ky_max
etalist = np.array(range(-Neta_half, Neta_half+1))*2.0*np.pi/keta_max


### save file
filename2 = 'Inverse_FT_Gauss_Nx=225_Neta=225.hdf5'
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


for step in range(Nstep):
    istep = step * 100
    print istep
    
    t0 = time()
    
    data = group['e_g_'+str(istep)][()]
    re_e_tilde = data[:,:,:,0]
    #im_e_tilde = data[:,:,:,1]
    #re_gx_tilde = data[:,:,:,2]
    #im_gx_tilde = data[:,:,:,3]
    #re_gy_tilde = data[:,:,:,4]
    #im_gy_tilde = data[:,:,:,5]
    #re_geta_tilde = data[:,:,:,6]
    #im_geta_tilde = data[:,:,:,7]
    
    t1 = time()
    print t1 - t0
    
    # extend to larger momentum space
    re_e_tilde = np.pad(re_e_tilde, ((N_add,N_add),(N_add,N_add),(N_add,N_add)), mode='constant', constant_values = 0)
    
    t2 = time()
    print t2 - t1
    
    re_e_tilde = np.fft.ifftshift(re_e_tilde)
    e_x = np.fft.ifftn(re_e_tilde) * factor
    e_x = np.fft.fftshift(e_x)
    
    t3 = time()
    print t3 - t2
    
    group2.create_dataset('re_e'+str(istep), data = e_x)
    
f.close()
f2.close()


