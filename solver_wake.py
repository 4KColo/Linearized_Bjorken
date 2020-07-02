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

def Eloss(Ein_jet, KappaSC, Temp, Tau):
    Xstop_sqd = ( 0.5/KappaSC * Ein_jet**(1./3) / Temp**(4./3) )**2
    Tau_sqd = Tau*Tau
    return 4./3.1416 * Ein_jet * Tau_sqd/Xstop_sqd / np.sqrt(Xstop_sqd-Tau_sqd)
    
def Gauss_Source(Kx, Ky, Keta, X0, Y0, Eta0, DE, Tau, Width_XY, Width_Eta):
    K2 = (Kx*Kx + Ky*Ky)*Width_XY*Width_XY + Keta*Keta*Width_Eta*Width_Eta
    Factor = DE/Tau * np.exp(-K2/2.)
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Eta0
    return Factor*np.exp(-1j*K_dot_X)

def Integrand_source(Tau, Y, Kx, Ky, Keta, X0, Y0, Eta0, Eta_over_S, Ein_jet, KappaSC, Width_para, Para_C1, Para_C2, Para_D):
    E_ = Y[0]
    Gx = Y[1]
    Gy = Y[2]
    Geta = Y[3]
    
    Temp_tau = Tau_to_Temp(Tau)
    Gamma_eta = Eta_over_S/Temp_tau
    Tau2 = Tau*Tau
    K_dot_G = Kx*Gx + Ky*Gy + Keta*Geta
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/Tau2
    
    dE = (1.+cs_sqd)/Tau*E_ + K_dot_G*1j
    
    dGx = Gx/Tau + cs_sqd*Kx*E_*1j + Gamma_eta*K2*Gx + Gamma_eta/3.*Kx*K_dot_G
    dGy = Gy/Tau + cs_sqd*Ky*E_*1j + Gamma_eta*K2*Gy + Gamma_eta/3.*Ky*K_dot_G
    
    dGeta = 3.*Geta/Tau + cs_sqd*Keta/Tau2*E_*1j + Gamma_eta*K2*Geta + Gamma_eta/(3.*Tau2)*Keta*K_dot_G
    
    X_tau = X0 + Para_D*(Tau - tau_i)
    dE_dtau = Eloss(Ein_jet, KappaSC, Temp_tau, Tau)
    WidthXY = Width_para/(3.1416*Temp_tau)
    WidthEta = Width_para/(1.*3.1416)
    factor_C = np.exp(-WidthEta*WidthEta/2.)
    C1 = dE_dtau * Para_C1 * factor_C
    C2 = dE_dtau * Para_C2 * factor_C
    D  = dE_dtau * Para_D
    dE -= Gauss_Source(Kx, Ky, Keta, X_tau, Y0, Eta0, C1, Tau, WidthXY, WidthEta)
    dGx -= Gauss_Source(Kx, Ky, Keta, X_tau, Y0, Eta0, D, Tau, WidthXY, WidthEta)
    dGeta -= Gauss_Source(Kx, Ky, Keta, X_tau, Y0, Eta0, C2, Tau, WidthXY, WidthEta)
    
    return -np.array([dE, dGx, dGy, dGeta])


def Integrand_free(Tau, Y, Kx, Ky, Keta, X0, Y0, Eta0, Eta_over_S):
    E_ = Y[0]
    Gx = Y[1]
    Gy = Y[2]
    Geta = Y[3]
    
    Temp_tau = Tau_to_Temp(Tau)
    Gamma_eta = Eta_over_S/Temp_tau
    Tau2 = Tau*Tau
    K_dot_G = Kx*Gx + Ky*Gy + Keta*Geta
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/Tau2
    
    dE = (1.+cs_sqd)/Tau*E_ + K_dot_G*1j
    
    dGx = Gx/Tau + cs_sqd*Kx*E_*1j + Gamma_eta*K2*Gx + Gamma_eta/3.*Kx*K_dot_G
    dGy = Gy/Tau + cs_sqd*Ky*E_*1j + Gamma_eta*K2*Gy + Gamma_eta/3.*Ky*K_dot_G
    
    dGeta = 3.*Geta/Tau + cs_sqd*Keta/Tau2*E_*1j + Gamma_eta*K2*Geta + Gamma_eta/(3.*Tau2)*Keta*K_dot_G
    
    return -np.array([dE, dGx, dGy, dGeta])


### parameters
x0 = -1.0/C1    # GeV^-1
y0 = 0.0
eta0 = 0.0
cosh_eta0 = np.cosh(eta0)
sinh_eta0 = np.sinh(eta0)
coeff_c1 = cosh_eta0 + sinh_eta0*sinh_eta0/cosh_eta0
coeff_c2 = -2. * sinh_eta0
coeff_d  = 1./cosh_eta0
crit_tau = 4.6/C1
kappa_sc = 0.4
E_in = 100.0            # initial jet energy, units: GeV
sigma = 1.0             # Width in the Gaussian source, units: 1/(pi T)
viscosity = 0.0         # eta/s in units of 1/4pi
viscosity_over_s = viscosity/(4.0*np.pi)

nstep = 1
#list_tau = np.linspace(25.0, tau_f, nstep)
list_tau = np.array([tau_f])
print(list_tau)

### parallel computing
Nx = 255
Ny = 255
Neta = 251
kx = np.linspace(-10.0, 10.0, Nx)        # GeV
ky = np.linspace(-10.0, 10.0, Ny)        # GeV
keta = np.linspace(-20.0, 20.0, Neta)    # unit 1

shared_array_base = mp.Array(ctypes.c_double, Nx*Ny*Neta*8*nstep)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(Nx,Ny,Neta,8,nstep)

def Integrator(X, List_e = shared_array):
    I_kx = X[0]
    I_ky = X[1]
    I_keta = X[2]
    Kx = kx[I_kx]
    Ky = ky[I_ky]
    Keta = keta[I_keta]
    Y_i = np.array([0.+0.*1j, 0., 0., 0.])
    a1 = si.solve_ivp(Integrand_source, (tau_i, crit_tau), Y_i, method='RK45', t_eval=[crit_tau],
    				args=(Kx, Ky, Keta, x0, y0, eta0, viscosity_over_s, E_in, kappa_sc, sigma, coeff_c1, coeff_c2, coeff_d))
    a2 = si.solve_ivp(Integrand_free, (crit_tau, tau_f), a1.y.flatten(), method='RK45', t_eval=list_tau, args=(Kx, Ky, Keta, x0, y0, eta0, viscosity_over_s))
    shared_array[I_kx, I_ky, I_keta, 0] = a2.y[0].real
    shared_array[I_kx, I_ky, I_keta, 1] = a2.y[0].imag
    shared_array[I_kx, I_ky, I_keta, 2] = a2.y[1].real
    shared_array[I_kx, I_ky, I_keta, 3] = a2.y[1].imag
    shared_array[I_kx, I_ky, I_keta, 4] = a2.y[2].real
    shared_array[I_kx, I_ky, I_keta, 5] = a2.y[2].imag
    shared_array[I_kx, I_ky, I_keta, 6] = a2.y[3].real
    shared_array[I_kx, I_ky, I_keta, 7] = a2.y[3].imag
    return None

t0 = time()
Nprocess = mp.cpu_count()
pool = mp.Pool(Nprocess)
pool.map(Integrator, [(i,j,k) for i in range(Nx) for j in range(Ny) for k in range(Neta)])
t1 = time()
print(t1-t0)

### save file
filename = 'SolverWake_Nx='+str(Nx)+'_Neta='+str(Neta)+'.hdf5'
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
group.create_dataset('e_and_g', data = shared_array)

f.close()

