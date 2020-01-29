#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py
import os

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif'})
rc('text', usetex=True)

import matplotlib as mpl
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 


viscosity = 1	# eta/s in units of 1/4pi
N = 100
xlist = np.linspace(-5.0, 5.0, N)
ylist = np.linspace(-5.0, 5.0, N)
etalist = np.linspace(-5.0, 5.0, N)
delta_e_re = np.empty([N,N,N])

def InverseFT_Real(X_, Y_, Eta_, dKx_, dKy_, dKeta_, Grid_Kx, Grid_Ky, Grid_Keta, Data):
	Cos = np.cos( X_*Grid_Kx + Y_*Grid_Ky + Eta_*Grid_Keta ).transpose(1,0,2)
	Real = np.sum(Cos * Data) * dKx_*dKy_*dKeta_/(2.*np.pi)**3
	return Real


def InverseFT_Imag(X_, Y_, Eta_, dKx_, dKy_, dKeta_, Grid_Kx, Grid_Ky, Grid_Keta, Data):
	Sin = np.sin( X_*Grid_Kx + Y_*Grid_Ky + Eta_*Grid_Keta ).transpose(1,0,2)
	Imag = np.sum(Sin * Data) * dKx_*dKy_*dKeta_/(2.*np.pi)**3
	return Imag


### read in file
filename = 'Linearized_Hydro_over_Bjorken.hdf5'
f = h5py.File(filename, 'r')
    
groupname = '4pi_eta_over_s='+str(viscosity)
if groupname in f:
    group = f[groupname]
    kx = group.attrs['kx']
    ky = group.attrs['ky']
    keta = group.attrs['keta']
    tau = group['tau'][()]

Nstep = len(tau)
d_kx = kx[1]-kx[0]
d_ky = ky[1]-ky[0]
d_keta = keta[1]-keta[0]
kx_grid, ky_grid, keta_grid = np.meshgrid(kx, ky, keta)

for istep in range(Nstep):
	data = group['e_g_'+str(istep)][()]
	re_e_tilde = data[:,:,:,0]
	im_e_tilde = data[:,:,:,1]
	re_gx_tilde = data[:,:,:,2]
	im_gx_tilde = data[:,:,:,3]
	re_gy_tilde = data[:,:,:,4]
	im_gy_tilde = data[:,:,:,5]
	re_geta_tilde = data[:,:,:,6]
	im_geta_tilde = data[:,:,:,7]
	
	for ix in range(N):
		x = xlist[ix]
		for iy in range(N):
			y = ylist[iy]
			for ieta in range(N):
				eta = etalist[ieta]
				delta_e_re[ix,iy,ieta] = InverseFT_Real(x, y, eta, d_kx, d_ky, d_keta, kx_grid, ky_grid, keta_grid, re_e_tilde)

f.close()
