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
    tau_fm = group['tau'][()]
    tau = tau_fm/C1
    
Nx = len(x)
Ny = len(y)
Neta = len(eta)
Nstep = int(len(tau)/100.)
d_x = x[1]-x[0]
d_y = y[1]-y[0]
d_eta = eta[1]-eta[0]
Ncut_xy = 50
Ncut_eta = 80
Nx_new1 = int((Nx-1)/2) - Ncut_xy
Ny_new1 = int((Ny-1)/2) - Ncut_xy
Neta_new1 = int((Neta-1)/2) - Ncut_eta
Nx_new2 = int((Nx+1)/2) + Ncut_xy
Ny_new2 = int((Ny+1)/2) + Ncut_xy
Neta_new2 = int((Neta+1)/2) + Ncut_eta
x = x[Nx_new1:Nx_new2]
y = y[Ny_new1:Ny_new2]
eta = eta[Neta_new1:Neta_new2]
x_grid, y_grid, eta_grid = np.meshgrid(y, x, eta)
x_2D, y_2D = np.meshgrid(x*C1, y*C1)
factor = d_x*d_y*d_eta

for step in range(Nstep):
    istep = step * 100
    epsilon = group['e'+str(istep)][()].real[Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
    geta = group['geta'+str(istep)][()].real[Nx_new1:Nx_new2, Ny_new1:Ny_new2, Neta_new1:Neta_new2]
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.locator_params(nbins=5)
    ax.plot_surface(x_2D, y_2D, epsilon[:,:,Ncut_eta], rstride=1, cstride=1, cmap='viridis')
    zbottom, ztop = ax.get_zlim3d()
    ax.set_xlabel(r'$x$ (fm)', size = 15)
    ax.set_ylabel(r'$y$ (fm)', size = 15)
    ax.set_zlabel(r'$\delta\epsilon$', size = 15)
    ax.text(-1, 2, ztop, r'$\delta \epsilon(x,y,\eta=0,\tau=$ '+str(tau_fm[istep])+' fm)', fontsize=15, color='red')
    plt.savefig('./plots/3Dplot_point_source_epsilon_eta=0_tau='+str(tau_fm[istep])+'fm.pdf')
    
    ax = plt.axes(projection='3d')
    plt.locator_params(nbins=5)
    ax.plot_surface(x_2D, y_2D, geta[:,:,Ncut_eta], rstride=1, cstride=1, cmap='viridis')
    zbottom, ztop = ax.get_zlim3d()
    ax.set_xlabel(r'$x$ (fm)', size = 15)
    ax.set_ylabel(r'$y$ (fm)', size = 15)
    ax.set_zlabel(r'$g_{\eta}$', size = 15)
    ax.text(-1, 2, ztop, r'$g_{\eta}(x,y,\eta=0,\tau=$ '+str(tau_fm[istep])+' fm)', fontsize=15, color='red')
    plt.savefig('./plots/3Dplot_point_source_geta_eta=0_tau='+str(tau_fm[istep])+'fm.pdf')
    '''
    epsilon_eta = np.sum(epsilon,axis=(0,1))*factor
    geta_eta = np.sum(geta,axis=(0,1))*factor
    
    plt.figure(1)
    if step == Nstep-1:
        plt.plot(eta/tau[istep], epsilon_eta, label=r'$\tau=$'+str(tau_fm[istep])+'fm')
        plt.xlabel(r'$\eta$', size = 15)
        plt.ylabel(r'$\tau d\eta \int dx\, \int dy\, \delta\epsilon$', size = 15)
        ybottom, ytop = plt.ylim()
        plt.xlim([-3.0,3.0])
        plt.legend(loc='best', fontsize=15)
        #plt.text(1.0, ytop*0.8, r'$\tau=$ '+str(tau_fm[istep])+' fm', fontsize=15)
        plt.savefig('./plots/int_epsilon_tau='+str(tau_fm[istep])+'fm.pdf')
    else:
        plt.plot(eta/tau[istep], epsilon_eta, label=r'$\tau=$'+str(tau_fm[istep])+'fm')
    
    plt.figure(2)
    plt.plot(eta/tau[istep], geta_eta)
    plt.xlabel(r'$\eta$', size = 15)
    plt.ylabel(r'$\tau d\eta\int dx\, \int dy\, g_{\eta}$', size = 15)
    ybottom, ytop = plt.ylim()
    plt.xlim([-3.0,3.0])
    plt.text(1.0, ytop*0.8, r'$\tau=$ '+str(tau_fm[istep])+' fm', fontsize=15)
    plt.savefig('./plots/int_geta_tau='+str(tau_fm[istep])+'fm.pdf')
    
    print(si.simps(epsilon_eta), si.simps(geta_eta))
    
    epsilon_eta = epsilon_eta*np.cosh(eta/tau[istep])           # T_ttau
    geta_eta = geta_eta*np.sinh(eta/tau[istep]) * tau[istep]    # T_etatau
    
    print(si.simps(epsilon_eta), si.simps(geta_eta))
    
    plt.figure(3)
    if step == Nstep-1:
        plt.plot(eta/tau[istep], epsilon_eta, label=r'$\tau=$'+str(tau_fm[istep])+'fm')
        plt.xlabel(r'$\eta$', size = 15)
        plt.ylabel(r'$\tau d\eta\int dx\, \int dy (\cosh\eta) \delta\epsilon $', size = 15)
        ybottom, ytop = plt.ylim()
        plt.xlim([-3.0,3.0])
        plt.legend(loc='best', fontsize=15)
        #plt.text(1.0, ytop*0.8, r'$\tau=$ '+str(tau_fm[istep])+' fm', fontsize=15)
        plt.savefig('./plots/int_T_ttau_tau='+str(tau_fm[istep])+'fm.pdf')
    else:
        plt.plot(eta/tau[istep], epsilon_eta, label=r'$\tau=$'+str(tau_fm[istep])+'fm')
    
    plt.figure(4)
    plt.plot(eta/tau[istep], geta_eta)
    plt.xlabel(r'$\eta$', size = 15)
    plt.ylabel(r'$\tau d\eta\int dx\, \int dy \tau (\sinh\eta) g_{\eta} $', size = 15)
    ybottom, ytop = plt.ylim()
    plt.xlim([-3.0,3.0])
    plt.text(1.0, ytop*0.8, r'$\tau=$ '+str(tau_fm[istep])+' fm', fontsize=15)
    plt.savefig('./plots/int_T_etatau_tau='+str(tau_fm[istep])+'fm.pdf')
    
    #re_e_mid_eta = sn.gaussian_filter(re_e_mid_eta, sigma=1.0, order=0)

f.close()

