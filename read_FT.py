#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
import h5py
import os
from mpl_toolkits import mplot3d

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
viscosity = 1	# eta/s in units of 1/4pi


### read in file
filename = 'Inverse_FT_Gauss_Nx=225_Neta=225.hdf5'
f = h5py.File(filename, 'r')
    
groupname = '4pi_eta_over_s='+str(viscosity)
if groupname in f:
    group = f[groupname]
    x = group.attrs['x']
    y = group.attrs['y']
    eta = group.attrs['eta']
    tau = group['tau'][()]


Nx = len(x)
Ny = len(y)
Neta = len(eta)
Nstep = int(len(tau)/100.)
d_x = x[1]-x[0]
d_y = y[1]-y[0]
d_eta = eta[1]-eta[0]
Ncut = 150
x = x[(Nx-1)/2 - Ncut:(Nx+1)/2 + Ncut]
y = y[(Ny-1)/2 - Ncut:(Ny+1)/2 + Ncut]
eta = eta[(Neta-1)/2 - Ncut:(Neta+1)/2 + Ncut]
#print eta[125:176]
#print eta[125:176].shape
x_grid, y_grid, eta_grid = np.meshgrid(y, x, eta)
x_2D, y_2D = np.meshgrid(x*C1, y*C1)
factor = d_x*d_y*d_eta*tau/C1

for step in range(Nstep):
    istep = step * 100
    re_e = group['re_e'+str(istep)][()].real[(Nx-1)/2 - Ncut:(Nx+1)/2 + Ncut, (Ny-1)/2 - Ncut:(Ny+1)/2 + Ncut, (Neta-1)/2 - Ncut:(Neta+1)/2 + Ncut]
    
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.locator_params(nbins=5)
    ax.plot_surface(x_2D, y_2D, re_e[:,:,Ncut], rstride=1, cstride=1, cmap='viridis')
    zbottom, ztop = ax.get_zlim3d()
    ax.set_xlabel(r'$x$ (fm)', size = 15)
    ax.set_ylabel(r'$y$ (fm)', size = 15)
    ax.set_zlabel(r'$\Re\delta\epsilon$', size = 15)
    ax.text(-1, 1, ztop, r'$\Re \delta \epsilon(x,y,\eta=0,\tau=$ '+str(tau[istep])+' fm)', fontsize=15, color='red')
    plt.savefig('3Dplot_point_source_re_e_eta=0_tau='+str(tau[istep])+'fm.pdf')
    #plt.show()
    '''
    re_e_eta = np.sum(re_e,axis=(0,1))*factor[istep]
    
    plt.figure()
    plt.plot(eta[105:196], re_e_eta[105:196])
    plt.xlabel(r'$\eta$', size = 15)
    plt.ylabel(r'$\int dx\, dy\, \Re\delta\epsilon$', size = 15)
    ybottom, ytop = plt.ylim()
    plt.text(2.0, ytop*0.8, r'$\tau=$ '+str(tau[istep])+' fm', fontsize=15)
    plt.savefig('./plots/int_re_e_tau='+str(tau[istep])+'fm.pdf')
    #plt.show()
    
    re_e_eta = re_e_eta*np.cosh(eta)    # T_ttau
    
    print si.simps(re_e_eta[105:196]), np.sum(re_e_eta[105:196])
    
    plt.figure()
    plt.plot(eta[105:196], re_e_eta[105:196])
    plt.xlabel(r'$\eta$', size = 15)
    plt.ylabel(r'$\int dx\, dy (\cosh\eta) \Re\delta\epsilon $', size = 15)
    ybottom, ytop = plt.ylim()
    plt.text(2.0, ytop*0.8, r'$\tau=$ '+str(tau[istep])+' fm', fontsize=15)
    plt.savefig('./plots/int_T_ttau_tau='+str(tau[istep])+'fm.pdf')
    #plt.show()
    
    #re_e_mid_eta = sn.gaussian_filter(re_e_mid_eta, sigma=1.0, order=0)

f.close()

