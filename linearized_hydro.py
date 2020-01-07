#!/usr/bin/env python

import numpy as np
import h5py
import os
import multiprocessing as mp
import ctypes
from time import time

cs_sqd = 1./3               # speed of sound squared
viscosity_over_s = 0.1      # eta/s
temp0 = 0.45                # initial temp of QGP
temp_c = 0.154              # critical temp of QGP

C1 = 0.197327               # 0.197327 GeV*fm = 1
tau_i = 0.6/C1              # convert to GeV^-1
tau_f = tau_i*(temp0/temp_c)**(1/cs_sqd)        # ideal Bjorken expansion
dtau = 0.01/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)
list_temp = temp0*(tau_i/list_tau)**cs_sqd
list_gamma_eta = viscosity_over_s/list_temp     # gamma_eta = eta/(e+P) = eta/s/T

def OneStep(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta,
            Kx, Ky, Keta, Tau, dTau, Cs2, Gamma_eta):
    Cs2plus1 = 1.+Cs2
    K_dot_Re = Kx*Re_gx + Ky*Re_gy + Keta*Re_geta
    K_dot_Im = Kx*Im_gx + Ky*Im_gy + Keta*Im_geta
    Tau2 = Tau*Tau
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/Tau2
    
    dRe_e  = Cs2plus1/Tau * Re_e - K_dot_Im
    dIm_e  = Cs2plus1/Tau * Im_e + K_dot_Re
    
    dRe_gx = Re_gx/Tau - Cs2*Kx*Im_e + Gamma_eta*K2*Re_gx + 1./3*Gamma_eta*Kx*K_dot_Re
    dIm_gx = Im_gx/Tau + Cs2*Kx*Re_e + Gamma_eta*K2*Im_gx + 1./3*Gamma_eta*Kx*K_dot_Im
    
    dRe_gy = Re_gy/Tau - Cs2*Ky*Im_e + Gamma_eta*K2*Re_gy + 1./3*Gamma_eta*Ky*K_dot_Re
    dIm_gy = Im_gy/Tau + Cs2*Ky*Re_e + Gamma_eta*K2*Im_gy + 1./3*Gamma_eta*Ky*K_dot_Im
    
    dRe_geta = 3.*Re_geta/Tau - Cs2*Keta*Im_e/Tau2 + Gamma_eta*K2*Re_geta + 1./(3.*Tau2)*Gamma_eta*Keta*K_dot_Re
    dIm_geta = 3.*Im_geta/Tau + Cs2*Keta*Re_e/Tau2 + Gamma_eta*K2*Im_geta + 1./(3.*Tau2)*Gamma_eta*Keta*K_dot_Im
    
    return -dRe_e*dTau, -dIm_e*dTau, -dRe_gx*dTau, -dIm_gx*dTau, -dRe_gy*dTau, -dIm_gy*dTau, -dRe_geta*dTau, -dIm_geta*dTau
    

def RK4(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta,
        Kx, Ky, Keta, Tau, dTau, Cs2, Gamma_eta):
    
    dRe_e1, dIm_e1, dRe_gx1, dIm_gx1, dRe_gy1, dIm_gy1, dRe_geta1, dIm_geta1 = OneStep(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta, Kx, Ky, Keta, Tau, dTau, Cs2, Gamma_eta)
    
    dRe_e2, dIm_e2, dRe_gx2, dIm_gx2, dRe_gy2, dIm_gy2, dRe_geta2, dIm_geta2 = OneStep(Re_e+dRe_e1/2., Im_e+dIm_e1/2., Re_gx+dRe_gx1/2., Im_gx+dIm_gx1/2., Re_gy+dRe_gy1/2., Im_gy+dIm_gy1/2., Re_geta+dRe_geta1/2., Im_geta+dIm_geta1/2., Kx, Ky, Keta, Tau+dTau/2., dTau, Cs2, Gamma_eta)
    
    dRe_e3, dIm_e3, dRe_gx3, dIm_gx3, dRe_gy3, dIm_gy3, dRe_geta3, dIm_geta3 = OneStep(Re_e+dRe_e2/2., Im_e+dIm_e2/2., Re_gx+dRe_gx2/2., Im_gx+dIm_gx2/2., Re_gy+dRe_gy2/2., Im_gy+dIm_gy2/2., Re_geta+dRe_geta2/2., Im_geta+dIm_geta2/2., Kx, Ky, Keta, Tau+dTau/2., dTau, Cs2, Gamma_eta)
    
    dRe_e4, dIm_e4, dRe_gx4, dIm_gx4, dRe_gy4, dIm_gy4, dRe_geta4, dIm_geta4 = OneStep(Re_e+dRe_e3, Im_e+dIm_e3, Re_gx+dRe_gx3, Im_gx+dIm_gx3, Re_gy+dRe_gy3, Im_gy+dIm_gy3, Re_geta+dRe_geta3, Im_geta+dIm_geta3, Kx, Ky, Keta, Tau+dTau, dTau, Cs2, Gamma_eta)
    
    Re_e_new  = Re_e  + (dRe_e1  + 2.*dRe_e2  + 2.*dRe_e3  + dRe_e4)/6.
    Im_e_new  = Im_e  + (dIm_e1  + 2.*dIm_e2  + 2.*dIm_e3  + dIm_e4)/6.
    Re_gx_new = Re_gx + (dRe_gx1 + 2.*dRe_gx2 + 2.*dRe_gx3 + dRe_gx4)/6.
    Im_gx_new = Im_gx + (dIm_gx1 + 2.*dIm_gx2 + 2.*dIm_gx3 + dIm_gx4)/6.
    Re_gy_new = Re_gy + (dRe_gy1 + 2.*dRe_gy2 + 2.*dRe_gy3 + dRe_gy4)/6.
    Im_gy_new = Im_gy + (dIm_gy1 + 2.*dIm_gy2 + 2.*dIm_gy3 + dIm_gy4)/6.
    Re_geta_new = Re_geta + (dRe_geta1 + 2.*dRe_geta2 + 2.*dRe_geta3 + dRe_geta4)/6.
    Im_geta_new = Im_geta + (dIm_geta1 + 2.*dIm_geta2 + 2.*dIm_geta3 + dIm_geta4)/6.
    
    return Re_e_new, Im_e_new, Re_gx_new, Im_gx_new, Re_gy_new, Im_gy_new, Re_geta_new, Im_geta_new


def Gauss_source(Kx, Ky, Keta, Tau, X0, Y0, Eta0, DE, Width):
    Sigma2 = Width * Width
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/(Tau*Tau)
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Eta0
    Same_factor = DE/Tau * np.exp(-0.5*K2*Sigma2)
    return Same_factor*np.cos(K_dot_X), -Same_factor*np.sin(K_dot_X)


def Linearized_hydro(Kx, Ky, Keta):
    # source for perturbations in each step
    re_e_source = np.zeros(nstep)
    im_e_source = np.zeros(nstep)
    re_gx_source = np.zeros(nstep)
    im_gx_source = np.zeros(nstep)
    re_gy_source = np.zeros(nstep)
    im_gy_source = np.zeros(nstep)
    re_geta_source = np.zeros(nstep)
    im_geta_source = np.zeros(nstep)
    re_e_source[0], im_e_source[0] = Gauss_source(Kx, Ky, Keta, tau_i, 0.0, 0.0, 0.0, 1.0, 0.2)
    
    re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = 0., 0., 0., 0., 0., 0., 0., 0.
    
    for i in range(nstep):
        re_e_old  = re_e_new  + re_e_source[i]
        im_e_old  = im_e_new  + im_e_source[i]
        re_gx_old = re_gx_new + re_gx_source[i]
        im_gx_old = im_gx_new + im_gx_source[i]
        re_gy_old = re_gy_new + re_gy_source[i]
        im_gy_old = im_gy_new + im_gy_source[i]
        re_geta_old = re_geta_new + re_geta_source[i]
        im_geta_old = im_geta_new + im_geta_source[i]
                        
        re_e_new, im_e_new, re_gx_new, im_gx_new, re_gy_new, im_gy_new, re_geta_new, im_geta_new = RK4(re_e_old, im_e_old, re_gx_old, im_gx_old, re_gy_old, im_gy_old, re_geta_old, im_geta_old, Kx, Ky, Keta, list_tau[i], dtau, cs_sqd, list_gamma_eta[i])
        
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
    Output[i][j][k] = Linearized_hydro(kx[i], ky[j], keta[k])
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

