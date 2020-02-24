#!/usr/bin/env python

import numpy as np
from constants import *

def Tau_to_Temp(Tau):
    return temp0*(tau_i/Tau)**cs_sqd

### evolution functions
def OneStep(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta,
            Kx, Ky, Keta, Tau, dTau, Cs2, Gamma_eta):
    Cs2plus1 = 1.+Cs2
    #K_dot_Re = Kx*Re_gx + Ky*Re_gy + Keta*Tau*Re_geta
    K_dot_Im = Kx*Im_gx + Ky*Im_gy + Keta*Tau*Im_geta
    #K2 = Kx*Kx + Ky*Ky + Keta*Keta
    #Tau2 = Tau*Tau
    
    
    dRe_e  = Cs2plus1/Tau * Re_e - K_dot_Im
    dIm_e  = 0.0#Cs2plus1/Tau * Im_e + K_dot_Re
    
    dRe_gx = 0.0#Re_gx/Tau - Cs2*Kx*Im_e + Gamma_eta*K2*Re_gx + 1./3*Gamma_eta*Kx*K_dot_Re
    dIm_gx = 0.0#Im_gx/Tau + Cs2*Kx*Re_e + Gamma_eta*K2*Im_gx + 1./3*Gamma_eta*Kx*K_dot_Im
    
    dRe_gy = 0.0#Re_gy/Tau - Cs2*Ky*Im_e + Gamma_eta*K2*Re_gy + 1./3*Gamma_eta*Ky*K_dot_Re
    dIm_gy = 0.0#Im_gy/Tau + Cs2*Ky*Re_e + Gamma_eta*K2*Im_gy + 1./3*Gamma_eta*Ky*K_dot_Im
    
    dRe_geta = 0.0#3.*Re_geta/Tau - Cs2*Keta*Im_e/Tau + Gamma_eta*K2*Re_geta + 1./(3.*Tau)*Gamma_eta*Keta*Tau*K_dot_Re
    dIm_geta = 0.0#3.*Im_geta/Tau + Cs2*Keta*Re_e/Tau + Gamma_eta*K2*Im_geta + 1./(3.*Tau)*Gamma_eta*Keta*Tau*K_dot_Im
    
    return -dRe_e*dTau, -dIm_e*dTau, -dRe_gx*dTau, -dIm_gx*dTau, -dRe_gy*dTau, -dIm_gy*dTau, -dRe_geta*dTau, -dIm_geta*dTau
    

def RK4(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta,
        Kx, Ky, Keta, Tau, dTau, Cs2, Viscosity_over_S):
    
    Tau_half = Tau + 0.5 * dTau
    Tau_one  = Tau + dTau
    
    Gamma_eta1  = Viscosity_over_S/Tau_to_Temp(Tau)         # gamma_eta = eta/(e+P) = eta/s/T
    Gamma_eta23 = Viscosity_over_S/Tau_to_Temp(Tau_half)
    Gamma_eta4  = Viscosity_over_S/Tau_to_Temp(Tau_one)
    
    dRe_e1, dIm_e1, dRe_gx1, dIm_gx1, dRe_gy1, dIm_gy1, dRe_geta1, dIm_geta1 = OneStep(Re_e, Im_e, Re_gx, Im_gx, Re_gy, Im_gy, Re_geta, Im_geta, Kx, Ky, Keta, Tau, dTau, Cs2, Gamma_eta1)
    
    dRe_e2, dIm_e2, dRe_gx2, dIm_gx2, dRe_gy2, dIm_gy2, dRe_geta2, dIm_geta2 = OneStep(Re_e+dRe_e1/2., Im_e+dIm_e1/2., Re_gx+dRe_gx1/2., Im_gx+dIm_gx1/2., Re_gy+dRe_gy1/2., Im_gy+dIm_gy1/2., Re_geta+dRe_geta1/2., Im_geta+dIm_geta1/2., Kx, Ky, Keta, Tau_half, dTau, Cs2, Gamma_eta23)
    
    dRe_e3, dIm_e3, dRe_gx3, dIm_gx3, dRe_gy3, dIm_gy3, dRe_geta3, dIm_geta3 = OneStep(Re_e+dRe_e2/2., Im_e+dIm_e2/2., Re_gx+dRe_gx2/2., Im_gx+dIm_gx2/2., Re_gy+dRe_gy2/2., Im_gy+dIm_gy2/2., Re_geta+dRe_geta2/2., Im_geta+dIm_geta2/2., Kx, Ky, Keta, Tau_half, dTau, Cs2, Gamma_eta23)
    
    dRe_e4, dIm_e4, dRe_gx4, dIm_gx4, dRe_gy4, dIm_gy4, dRe_geta4, dIm_geta4 = OneStep(Re_e+dRe_e3, Im_e+dIm_e3, Re_gx+dRe_gx3, Im_gx+dIm_gx3, Re_gy+dRe_gy3, Im_gy+dIm_gy3, Re_geta+dRe_geta3, Im_geta+dIm_geta3, Kx, Ky, Keta, Tau_one, dTau, Cs2, Gamma_eta4)
    
    Re_e_new  = Re_e  + (dRe_e1  + 2.*dRe_e2  + 2.*dRe_e3  + dRe_e4)/6.
    Im_e_new  = Im_e  + (dIm_e1  + 2.*dIm_e2  + 2.*dIm_e3  + dIm_e4)/6.
    Re_gx_new = Re_gx + (dRe_gx1 + 2.*dRe_gx2 + 2.*dRe_gx3 + dRe_gx4)/6.
    Im_gx_new = Im_gx + (dIm_gx1 + 2.*dIm_gx2 + 2.*dIm_gx3 + dIm_gx4)/6.
    Re_gy_new = Re_gy + (dRe_gy1 + 2.*dRe_gy2 + 2.*dRe_gy3 + dRe_gy4)/6.
    Im_gy_new = Im_gy + (dIm_gy1 + 2.*dIm_gy2 + 2.*dIm_gy3 + dIm_gy4)/6.
    Re_geta_new = Re_geta + (dRe_geta1 + 2.*dRe_geta2 + 2.*dRe_geta3 + dRe_geta4)/6.
    Im_geta_new = Im_geta + (dIm_geta1 + 2.*dIm_geta2 + 2.*dIm_geta3 + dIm_geta4)/6.
    
    return Re_e_new, Im_e_new, Re_gx_new, Im_gx_new, Re_gy_new, Im_gy_new, Re_geta_new, Im_geta_new

### energy loss formula and source functions
def XStop(Ein, T, Ksc):
    return Ein**(1./3)/T**(4./3)/(2.*Ksc)
    
def ELoss(Ein, T, Ksc, X):
    _xstop = XStop(Ein, T, Ksc)
    _xstop2 = _xstop*_xstop
    _x2 = X*X
    if _xstop2 > _x2:
        return 4./np.pi * Ein * _x2/_xstop2 / np.sqrt(_xstop2-_x2)
    else:
        return 0.0
        
def Gauss_Source(Kx, Ky, Keta, Tau, dTau, X0, Y0, Eta0, Ein, T, Ksc, WidthPara):
    # X0, Y0, Eta0 are the current spatial positions of the jet
    DE = ELoss(Ein, T, Ksc, Tau - tau_i)*dTau     # energy loss during dTau
    Width = WidthPara/(np.pi*T)
    Sigma2 = Width * Width
    K2 = Kx*Kx + Ky*Ky + Keta*Keta/(Tau*Tau)
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Eta0
    Same_factor = DE/Tau * np.exp(-0.5*K2*Sigma2)
    return Same_factor*np.cos(K_dot_X), -Same_factor*np.sin(K_dot_X)

def Point_Source(Kx, Ky, Keta, Tau, dTau, X0, Y0, Eta0, Ein, T, Ksc):
    # X0, Y0, Eta0 are the current spatial positions of the jet
    DE = ELoss(Ein, T, Ksc, Tau - tau_i)*dTau     # energy loss during dTau
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Eta0
    Same_factor = DE/Tau
    return Same_factor*np.cos(K_dot_X), -Same_factor*np.sin(K_dot_X)
    
def Point_Source_Test(Kx, Ky, Keta, X0, Y0, Eta0, DE, Tau):
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Tau*Eta0
    return DE*np.cos(K_dot_X), -DE*np.sin(K_dot_X)

def Gauss_Source_Test(Kx, Ky, Keta, X0, Y0, Eta0, DE, Tau, Width):
    K2 = Kx*Kx + Ky*Ky + Keta*Keta
    Factor = DE * np.exp(-K2*Width*Width/2.)
    K_dot_X = Kx*X0 + Ky*Y0 + Keta*Tau*Eta0
    return Factor*np.cos(K_dot_X), -Factor*np.sin(K_dot_X)
