#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py
import os

viscosity_over_s = 0.1

### read in file
filename = 'Linearized_Hydro_over_Bjorken.hdf5'
f = h5py.File(filename, 'r')
    
groupname = 'eta_over_s='+str(viscosity_over_s)
if groupname in f:
    group = f[groupname]
    N = group.attrs['N']
    kx = group.attrs['kx']
    ky = group.attrs['ky']
    keta = group.attrs['keta']
    data = group['e_g'][()]

f.close()

d_kx = kx[1]-kx[0]
d_ky = ky[1]-ky[0]
d_keta = keta[1]-keta[0]

### in momentum space
re_e_tilde = data[:,:,:,0]
im_e_tilde = data[:,:,:,1]
re_gx_tilde = data[:,:,:,2]
im_gx_tilde = data[:,:,:,3]
re_gy_tilde = data[:,:,:,4]
im_gy_tilde = data[:,:,:,5]
re_geta_tilde = data[:,:,:,6]
im_geta_tilde = data[:,:,:,7]


### in position space
ky_grid, kx_grid, keta_grid = np.meshgrid(kx, ky, keta)
def InverseFT(X, Y, Eta, Table_re, Table_im):
    Cos = np.cos(ky_grid*X + ky_grid*Y + keta_grid*Eta)
    Sin = np.sin(ky_grid*X + ky_grid*Y + keta_grid*Eta)
    
    return np.sum(Cos*Table_re - Sin*Table_im) *d_kx*d_ky*d_keta/(2.*np.pi)**3
    
print InverseFT(0.0, 0.0, 0.0, re_e_tilde, im_e_tilde)
