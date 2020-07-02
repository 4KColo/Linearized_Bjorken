#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py
import os
from constants import *
from time import time
import multiprocessing as mp
from functools import partial
import ctypes

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif'})
rc('text', usetex=True)

import matplotlib as mpl
label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

C1 = 0.197327
viscosity = 0	# eta/s in units of 1/4pi
viscosity_over_s = viscosity/(4.0*np.pi)
dof = 40.0

### read in file
filename = 'Inverse_FT_Wake_shortrun_vis0.hdf5'
f = h5py.File(filename, 'r')
    
groupname = '4pi_eta_over_s='+str(viscosity)
if groupname in f:
    group = f[groupname]
    x = group.attrs['x']
    y = group.attrs['y']
    eta = group.attrs['eta']
    tau = group['tau'][()]
    tau_fm = tau*C1

Nx = len(x)
Ny = len(y)
Neta = len(eta)
Nstep = 3
d_x = x[1]-x[0]
d_y = y[1]-y[0]
d_eta = eta[1]-eta[0]
Ncut_x = 126
Ncut_y = 126
Ncut_eta = 65
Nx_new1 = int((Nx-1)/2) - Ncut_x
Ny_new1 = int((Ny-1)/2) - Ncut_y
Neta_new1 = int((Neta-1)/2) - Ncut_eta
Nx_new2 = int((Nx+1)/2) + Ncut_x
Ny_new2 = int((Ny+1)/2) + Ncut_y
Neta_new2 = int((Neta+1)/2) + Ncut_eta
x = x[Nx_new1:Nx_new2]
y = y[Ny_new1:Ny_new2]
eta = eta[Neta_new1:Neta_new2]
x_grid, y_grid, eta_grid = np.meshgrid(x, y, eta, indexing='ij')
prefactor2 = d_x*d_y*d_eta*tau[Nstep-1] / (2.*3.1416)**3
print(tau[Nstep-1])

#t0 = time()
epsilon = group['e_g0 '+str(Nstep-1)][()][Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
gx = group['e_g1 '+str(Nstep-1)][()][Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
gy = group['e_g2 '+str(Nstep-1)][()][Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
geta = group['e_g3 '+str(Nstep-1)][()][Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
f.close()
#t1 = time()
#print(t1-t0)


### background Bjorken in the end
def Tau_to_Temp(Tau):
    return temp0*(tau_i/Tau)**cs_sqd

Tc = Tau_to_Temp(tau[Nstep-1])
print(Tc)
gamma_eta = viscosity_over_s/Tc
T_s0 = 4.*dof/3.1416**2 * Tc**4      # epsilon0 + P0 = T * s0
epsilon0 = T_s0*3./4.
prefactor1 = 3.1416**2/(3.*dof)

print(np.max(np.abs(epsilon))/epsilon0, np.min(np.abs(epsilon))/epsilon0)
print(np.max(np.abs(gx))/T_s0, np.min(np.abs(gx))/T_s0)
print(np.max(np.abs(gy))/T_s0, np.min(np.abs(gy))/T_s0)
print(np.max(np.abs(geta))/T_s0, np.min(np.abs(geta))/T_s0)

def nB(x):
	return np.exp(-x)/(1.-np.exp(-x))

def nF(x):
	return np.exp(-x)/(1.+np.exp(-x))
	
def Spectrum2(Pt, Phi, Y, M):
	# this one is slow, Boltzmann distribution
    MT = np.sqrt(Pt*Pt+M*M)
    Cosh = np.cosh(Y-eta_grid)
    Sinh = np.sinh(Y-eta_grid)
    Fac1 = MT * Cosh
    Fac2 = Fac1 - ( gx*Pt*np.cos(Phi)+gy*Pt*np.sin(Phi) + tau[Nstep-1]*geta*MT*Sinh )/T_s0
    Tc_new = ( 3.1416**2/(3.*dof) * (epsilon0+epsilon) )**0.25
    Fac3 = np.exp(-Fac2/Tc_new) - np.exp(-Fac1/Tc)
    return np.sum(Fac1*Fac3) * prefactor2

def Spectrum(Pt, Phi, Y, M):
    MT = np.sqrt(Pt*Pt+M*M)
    M_cosh = MT * np.cosh(Y-eta_grid)
    Fac = M_cosh - ( gx*Pt*np.cos(Phi) + gy*Pt*np.sin(Phi) + tau[Nstep-1]*geta*MT*np.sinh(Y-eta_grid) )/T_s0
    Fac *= ( prefactor1 * (epsilon0+epsilon) )**(-0.25)
    Fac = nB(Fac)
    Fac -= nB(M_cosh/Tc)
    return np.sum(M_cosh*Fac) * prefactor2

'''
t2 = time()
print(Spectrum2(1.0, 1.0, 0.0, 0.135))
t3 = time()
print(t3-t2)

t4 = time()
print(Spectrum(1.0, 1.0, 0.0, 0.135))
t5 = time()
print(t5-t4)

'''
### store spectrum
mass  = 0.138
N_pT  = 80
N_phi = 30
N_y   = 81
l_pT  = np.linspace(0.0, 5.0, N_pT)
l_phi = np.linspace(0.0, 2.*np.pi, N_phi)
l_y   = np.linspace(-3.6, 3.6, N_y)

shared_array_base = mp.Array(ctypes.c_double, N_pT*N_phi*N_y)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(N_pT,N_phi,N_y)

def Spectrum_para(X, List = shared_array):
	I_pT  = X[0]
	I_phi = X[1]
	I_y   = X[2]
	Pt    = l_pT[I_pT]
	Phi   = l_phi[I_phi]
	Y     = l_y[I_y]
	shared_array[I_pT, I_phi, I_y] = Spectrum(Pt, Phi, Y, mass)
	return None

Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
t6 = time()
pool.map(Spectrum_para, [(i,j,k) for i in range(N_pT) for j in range(N_phi) for k in range(N_y)])
t7 = time()
print(t7-t6)


### save file
filename = 'WakeSpectrum_shortrun_vis0.hdf5'
if not os.path.exists(filename):
    f = h5py.File(filename, 'w')
else:
    f = h5py.File(filename, 'a')
    
groupname = 'pion'
if groupname in f:
    del f[groupname]
group = f.create_group(groupname)

group.create_dataset('pT', data = l_pT)
group.create_dataset('phi', data = l_phi)
group.create_dataset('y', data = l_y)
group.create_dataset('spectrum', data = shared_array)

f.close()

