#!/usr/bin/env python

# compare analytic and numerical solutions with eta/s = 0 and kappa = 0

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

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

'''
### test effect of finite grid size on FT and inverse FT
N = 201
list_k = np.linspace(-5.0,5.0,N)
d_k = list_k[1]-list_k[0]
f_k = np.exp(-list_k*list_k/2.)
list_x = np.linspace(-10.0,10.0,N)
f_x = []
f_x_simps = []
for ix in list_x:
	f_x.append( np.sum(np.cos(list_k*ix)*f_k)*d_k/2./3.1416 )
	f_x_simps.append( si.simps(np.cos(list_k*ix)*f_k)*d_k/2./3.1416 )

f_x = np.array(f_x)
f_x_simps = np.array(f_x_simps)
plt.figure()
plt.plot(list_x, f_x_simps, linewidth=2.0)
plt.plot(list_x, f_x, linewidth=2.0, linestyle='--')
plt.yscale('log')
plt.show()
'''

'''
### test numpy.fft
n = 100
N = n+n+1
Lhalf = 20.0		# make the length long enough so we have enough resolution in momentum space
L = Lhalf*2
list_x = np.linspace(-Lhalf,Lhalf,N)

list_k = np.linspace(-5.0,5.0,N)	# just for plotting the analytic solution
f_k = np.exp(-list_k**2/2.)			# this is the analytic solution we want to compare with

# the following lines just want to show setting f(x)=0 outside some range is fine
# this allows us to extend the function to a larger region to gain resolution in momentum space
f_x = np.exp(-list_x**2/2.)/np.sqrt(2.*np.pi)
f_x_v2 = []
for each_x in list_x:
	if np.abs(each_x) <= 5.0:
		f_x_v2.append(np.exp(-each_x**2/2.)/np.sqrt(2.*np.pi))
	else:
		f_x_v2.append(0.0)
f_x_v2 = np.array(f_x_v2)

f_x_shift = np.fft.ifftshift(f_x_v2)	# use f_x_v2 or f_x to see the difference, shift 0 to the first element

ft_x = np.array(range(-n,n+1)) * 2.0*np.pi/L				# this is the momentum space grid: 2pi/L * k
ft_f_x = np.fft.fftshift( np.fft.fft(f_x_shift) ) * L/N		# FT and then shift the zero frequency entry to the center

plt.figure()
plt.plot(list_k, f_k, linewidth = 2.0, color='black')
plt.plot(ft_x, ft_f_x, linewidth = 2.0, linestyle='--', color='red')
plt.xlim([-5.,5.])
plt.show()
'''
'''
### test inverse FFT
n = 100
N = n+n+1
pmaxhalf = 30.0
pmax = pmaxhalf*2
list_k = np.linspace(-pmaxhalf,pmaxhalf,N)
f_k = np.exp(-list_k**2/2.)

list_x = np.linspace(-5.0, 5.0, N)		# just for plotting the analytic solution
f_x = np.exp(-list_x**2/2.)/np.sqrt(2.*np.pi)	# this is the analytic solution we want to compare with

f_k_shift = np.fft.ifftshift(f_k)									# shift zero momentum to the first entry
ift_f_k = np.fft.fftshift(np.fft.ifft(f_k_shift))*pmax/(2.*np.pi)	# Ft and then shift x=0 to the center
ift_k = np.array(range(-n,n+1)) * 2.*np.pi/pmax

plt.figure()
plt.plot(list_x, f_x, linewidth = 2.0, color='black')
plt.plot(ift_k, ift_f_k, linewidth = 2.0, linestyle='--', color='red')
plt.xlim([-5.,5.])
plt.show()
'''
'''
### test 2D inverse FFT
n = 100
N = n+n+1
pmaxhalf = 30.0
pmax = pmaxhalf*2
list_kx = np.linspace(-pmaxhalf,pmaxhalf,N)
list_ky = np.linspace(-pmaxhalf,pmaxhalf,N)
grid_kx, grid_ky = np.meshgrid(list_kx, list_ky)
f_k = np.exp(-(grid_kx**2+grid_ky**2)/2.)

list_x = np.linspace(-5.0, 5.0, N)		# just for plotting the analytic solution
f_x = np.exp(-list_x**2/2.)/np.sqrt(2.*np.pi)	# this is the analytic solution we want to compare with

f_k_shift = np.fft.ifftshift(f_k)										# shift zero momentum to the first entry
ift_f_k = np.fft.fftshift(np.fft.ifft2(f_k_shift))*(pmax/(2.*np.pi))**2	# Ft and then shift x=0 to the center
ift_kx = np.array(range(-n,n+1)) * 2.*np.pi/pmax
ift_ky = np.array(range(-n,n+1)) * 2.*np.pi/pmax


plt.figure()
plt.plot(list_x, f_x, linewidth = 2.0, color='black')
plt.plot(ift_kx, np.sum(ift_f_k,axis=0).real*(ift_ky[1]-ift_ky[0]), linewidth = 2.0, linestyle='--', color='red')
plt.xlim([-5.,5.])
plt.show()
'''

### test 3D inverse FFT
n1 = 200
N1 = n1+n1+1
n2 = 200
N2 = n2+n2+1
n3 = 200
N3 = n3+n3+1
pmaxhalf1 = 30.0
pmax1 = pmaxhalf1*2
pmaxhalf2 = 30.0
pmax2 = pmaxhalf2*2
pmaxhalf3 = 30.0
pmax3 = pmaxhalf3*2
list_k1 = np.linspace(-pmaxhalf1,pmaxhalf1,N1)
list_k2 = np.linspace(-pmaxhalf2,pmaxhalf2,N2)
list_k3 = np.linspace(-pmaxhalf3,pmaxhalf3,N3)
grid_k1, grid_k2, grid_k3 = np.meshgrid(list_k1, list_k2, list_k3)
f_k = np.exp(-(grid_k1**2+grid_k2**2+grid_k3**2)/2.).transpose(1,0,2)

list_x = np.linspace(-5.0, 5.0, N1)				# just for plotting the analytic solution
f_x = np.exp(-list_x**2/2.)/np.sqrt(2.*np.pi)	# this is the analytic solution we want to compare with

f_k_shift = np.fft.ifftshift(f_k)				# shift zero momentum to the first entry
ift_f_k = np.fft.fftshift(np.fft.ifftn(f_k_shift))*(pmax1/(2.*np.pi))*(pmax2/(2.*np.pi))*(pmax3/(2.*np.pi))	# Ft and then shift x=0 to the center
ift_k1 = np.array(range(-n1,n1+1)) * 2.*np.pi/pmax1
ift_k2 = np.array(range(-n2,n2+1)) * 2.*np.pi/pmax2
ift_k3 = np.array(range(-n3,n3+1)) * 2.*np.pi/pmax3


plt.figure()
plt.plot(list_x, f_x, linewidth = 2.0, color='black')
#plt.plot(ift_k1, np.sum(ift_f_k,axis=(1,2)).real*(ift_k2[1]-ift_k2[0])*(ift_k3[1]-ift_k3[0]), linewidth = 2.0, linestyle='--', color='red')
plt.plot(ift_k2, np.sum(ift_f_k,axis=(0,2)).real*(ift_k1[1]-ift_k1[0])*(ift_k3[1]-ift_k3[0]), linewidth = 2.0, linestyle='--', color='red')
#plt.plot(ift_k3, np.sum(ift_f_k,axis=(0,1)).real*(ift_k1[1]-ift_k1[0])*(ift_k2[1]-ift_k2[0]), linewidth = 2.0, linestyle='--', color='red')
plt.xlim([-5.,5.])
plt.show()
