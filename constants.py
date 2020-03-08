#!/usr/bin/env python
C1 = 0.197327               # 0.197327 GeV*fm = 1
cs_sqd = 1./3               # speed of sound squared
temp0 = 0.45                # initial temp of QGP
temp_c = 0.31               # critical temp of QGP
tau_i = 0.6/C1              # convert to GeV^-1
tau_f = tau_i*(temp0/temp_c)**(1/cs_sqd)        # ideal Bjorken expansion

#print tau_f*C1
