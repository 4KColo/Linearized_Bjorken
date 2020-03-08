import numpy as np
import scipy.integrate as si
from constants import *
import h5py
import os
import multiprocessing as mp
from functools import partial
import ctypes
from time import time


def Tau_to_Temp(Tau):
    return temp0*(tau_i/Tau)**cs_sqd

def Gauss_Source_Test(Kx, Ky, Keta, X0, Y0, Eta0, DE, Tau, Width):
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/(Tau*Tau)
    Factor = DE/Tau * np.exp(-K2*Width*Width/2.)
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Eta0
    return Factor*np.exp(-1j*K_dot_X)

def Integrand(Tau, Y, Kx, Ky, Keta, Eta_over_S):
    E_ = Y[0]
    Gx = Y[1]
    Gy = Y[2]
    Geta = Y[3]
    
    Gamma_eta = Eta_over_S/Tau_to_Temp(Tau)
    Tau2 = Tau*Tau
    K_dot_G = Kx*Gx + Ky*Gy + Keta*Geta
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/Tau2
    
    dE = (1.+cs_sqd)/Tau*E_ + K_dot_G*1j
    
    dGx = Gx/Tau + cs_sqd*Kx*E_*1j + Gamma_eta*K2*Gx + Gamma_eta/3.*Kx*K_dot_G
    dGy = Gy/Tau + cs_sqd*Ky*E_*1j + Gamma_eta*K2*Gy + Gamma_eta/3.*Ky*K_dot_G
    
    dGeta = 3.*Geta/Tau + cs_sqd*Keta/Tau2*E_*1j + Gamma_eta*K2*Geta + Gamma_eta/(3.*Tau2)*Keta*K_dot_G
    
    return -np.array([dE, dGx, dGy, dGeta])


### parameters
x0 = 0.0    # GeV^-1
y0 = 0.0
eta0 = 0.0
delta_E = 50.0          # Delta E, units: GeV
sigma = 1.0             # Width in the Gaussian source, units: GeV^-1
viscosity = 1.0            # eta/s in units of 1/4pi
viscosity_over_s = viscosity/(4.0*np.pi)

dtau = 0.3/C1
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(0.6, 1.8, 5)/C1
print(tau_f, list_tau)
### parallel computing
Nx = 201
Ny = 201
Neta = 201
kx = np.linspace(-5.0, 5.0, Nx)        # GeV
ky = np.linspace(-5.0, 5.0, Ny)        # GeV
keta = np.linspace(-5.0, 5.0, Neta)    # unit 1

shared_array_base = mp.Array(ctypes.c_double, Nx*Ny*Neta*2*nstep)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(Nx,Ny,Neta,2,nstep)

def Integrator(X, List_e = shared_array):
    I_kx = X[0]
    I_ky = X[1]
    I_keta = X[2]
    Kx = kx[I_kx]
    Ky = ky[I_ky]
    Keta = keta[I_keta]
    Epsilon0 = Gauss_Source_Test(Kx, Ky, Keta, x0, y0, eta0, delta_E, tau_i, sigma)
    Y_i = np.array([Epsilon0, 0., 0., 0.])
    a = si.solve_ivp(Integrand, (tau_i, tau_f), Y_i, method='RK45', t_eval=list_tau, args=(Kx, Ky, Keta, viscosity_over_s))
    shared_array[I_kx, I_ky, I_keta, 0] = a.y[0].real
    shared_array[I_kx, I_ky, I_keta, 1] = a.y[0].imag
    return None

Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
pool.map(Integrator, [(i,j,k) for i in range(Nx) for j in range(Ny) for k in range(Neta)])

### save file
filename = 'SolverTest_Gauss_Nx='+str(Nx)+'_Neta='+str(Neta)+'.hdf5'
if not os.path.exists(filename):
    f = h5py.File(filename, 'w')
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
group.create_dataset('epsilon', data = shared_array)

f.close()
