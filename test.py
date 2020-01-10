#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss
import cmath as cm
from evolution import Tau_to_Temp
from evolution import RK4
from evolution import Point_Source_Test
from constants import *


kx = 1.0    # GeV
ky = 1.0
keta = 0.0
x0 = 0.1    # GeV^-1
y0 = 0.1
eta0 = 0.0
delta_E = 10.0
viscosity_over_s = 0.0      # eta/s

dtau = 0.01/C1              # time step
nstep = int((tau_f-tau_i)/dtau)+1
list_tau = np.linspace(tau_i, tau_f, nstep)
list_temp = Tau_to_Temp(list_tau)

re_e = np.zeros(nstep)
im_e = np.zeros(nstep)
re_gx = np.zeros(nstep)
im_gx = np.zeros(nstep)
re_gy = np.zeros(nstep)
im_gy = np.zeros(nstep)
re_geta = np.zeros(nstep)
im_geta = np.zeros(nstep)

re_e[0], im_e[0] = Point_Source_Test(kx, ky, keta, x0, y0, eta0, delta_E)
re_gx[0], im_gx[0] = 0.0, 0.0
re_gy[0], im_gy[0] = 0.0, 0.0
re_geta[0], im_geta[0] = 0.0, 0.0

for i in range(nstep-1):
    re_e[i+1], im_e[i+1], re_gx[i+1], im_gx[i+1], re_gy[i+1], im_gy[i+1], re_geta[i+1], im_geta[i+1] = RK4(re_e[i], im_e[i], re_gx[i], im_gx[i], re_gy[i], im_gy[i], re_geta[i], im_geta[i], kx, ky, keta, list_tau[i], dtau, cs_sqd, viscosity_over_s)


### compare with analytic solution
analytic_re = []
analytic_im = []
for i in range(nstep):
    Tau = list_tau[i]
    result = Tau**(-2./3)*(29.3328 - 5.94606*1j)*( ss.jv(2./3, 0.816497*Tau) + 0.949828*ss.yv(2./3, 0.816497*Tau) )
    analytic_re.append(result.real)
    analytic_im.append(result.imag)

analytic_re = np.array(analytic_re)
analytic_im = np.array(analytic_im)

plt.figure()
plt.plot(list_tau, re_e, linewidth = 2.0, color='black')
plt.plot(list_tau, analytic_re, linewidth = 2.0, color='red', linestyle = '--')
plt.show()

