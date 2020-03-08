#!/usr/bin/env python

# compare analytic and numerical solutions with eta/s = 0 and kappa = 0

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import cmath as cm
from evolution import Tau_to_Temp
from evolution import RK4
from evolution import Point_Source_Test
from constants import *

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

kx = 5.0    # GeV
ky = 5.0
keta = 0.0
x0 = 0.0    # GeV^-1
y0 = 0.0
eta0 = 0.0
delta_E = 10.0      # Delta E, units: GeV
viscosity_over_s_list = [0.0, 1./(4*np.pi), 2./(4*np.pi)]	# eta/s

a = 2.+cs_sqd				# for analytic solution
b = cs_sqd*(kx*kx + ky*ky)

dtau = 0.003/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)
list_temp = Tau_to_Temp(list_tau)

### analytic solution
def A11(x_, a_, b_):
	return x_**((1.-a_)/2.)*ss.jv((a_-1.)/2., np.sqrt(b_)*x_)
	
def A12(x_, a_, b_):
	return x_**((1.-a_)/2.)*ss.yv((a_-1.)/2., np.sqrt(b_)*x_)
	
def A21(x_, a_, b_):
	b_sqrt = np.sqrt(b_)
	term1 = ((1.-a_)/2.+1.+cs_sqd)*x_**((1.-a_)/2.-1.)*ss.jv((a_-1.)/2., b_sqrt*x_)
	term2 = b_sqrt/2. * x_**((1.-a_)/2.) * ( ss.jv((a_-1.)/2.-1., b_sqrt*x_) - ss.jv((a_-1.)/2.+1., b_sqrt*x_) )
	return term1+term2

def A22(x_, a_, b_):
	b_sqrt = np.sqrt(b_)
	term1 = ((1.-a_)/2.+1.+cs_sqd)*x_**((1.-a_)/2.-1.)*ss.yv((a_-1.)/2., b_sqrt*x_)
	term2 = b_sqrt/2. * x_**((1.-a_)/2.) * ( ss.yv((a_-1.)/2.-1., b_sqrt*x_) - ss.yv((a_-1.)/2.+1., b_sqrt*x_) )
	return term1+term2

matrix_A = np.array([[A11(tau_i, a, b), A12(tau_i, a, b)],[A21(tau_i, a, b), A22(tau_i, a, b)]])
matrix_B = np.array([delta_E/tau_i, 0.0])
coeff_analytic = np.linalg.solve(matrix_A, matrix_B)
sol_analytic = coeff_analytic[0]*A11(list_tau, a, b) + coeff_analytic[1]*A12(list_tau, a, b)


### comparison plots
plt.figure()
plt.plot(list_tau*C1, sol_analytic, linewidth = 2.0, color='black', label='analytic, $4\pi\eta/s$=0')

### numerical solutions
for j in range(len(viscosity_over_s_list)):
	viscosity_over_s = viscosity_over_s_list[j]
	re_e = np.zeros(nstep)
	im_e = np.zeros(nstep)
	re_gx = np.zeros(nstep)
	im_gx = np.zeros(nstep)
	re_gy = np.zeros(nstep)
	im_gy = np.zeros(nstep)
	re_geta = np.zeros(nstep)
	im_geta = np.zeros(nstep)

	re_e[0], im_e[0] = Point_Source_Test(kx, ky, keta, x0, y0, eta0, delta_E, tau_i)
	re_gx[0], im_gx[0] = 0.0, 0.0
	re_gy[0], im_gy[0] = 0.0, 0.0
	re_geta[0], im_geta[0] = 0.0, 0.0
	
	for i in range(nstep-1):
		re_e[i+1], im_e[i+1], re_gx[i+1], im_gx[i+1], re_gy[i+1], im_gy[i+1], re_geta[i+1], im_geta[i+1] = RK4(re_e[i], im_e[i], re_gx[i], im_gx[i], re_gy[i], im_gy[i], re_geta[i], im_geta[i], kx, ky, keta, list_tau[i], dtau, cs_sqd, viscosity_over_s)
	if j == 0:
		plt.plot(list_tau*C1, re_e, linewidth = 2.0, color='red', linestyle='--', label=r'numerical, $4\pi\eta/s$=0')
	if j == 1:
		plt.plot(list_tau*C1, re_e, linewidth = 2.0, color='blue', label=r'numerical, $4\pi\eta/s$='+str(j))
	if j == 2:
		plt.plot(list_tau*C1, re_e, linewidth = 2.0, color='green', label=r'numerical, $4\pi\eta/s$='+str(j))
		
plt.xlabel(r'$\tau$(fm/c)', size = 20)
plt.ylabel(r'$\Re\delta\tilde{\epsilon}$(GeV$^2$)', size = 20)
plt.legend(loc='best', fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
#plt.savefig('test_Re_e_kx='+str(kx)+'_ky='+str(ky)+'_x0='+str(x0)+'_y0='+str(y0)+'.pdf')
plt.show()
