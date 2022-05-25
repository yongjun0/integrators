'''
Simple Harmonic osc.
m = 1
omega = 1

x = cos
v = -sin
a = -cos = -x
H = 0.5(v**2 + x**2)
compare the err between the analytic sol and the integrator sols.
'''
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys

import sub_error as sub


x0 = 0.45 ### keep!

x0 = 0.52
p0 = -np.sqrt(1-x0**2)
a0 = -x0
pva = [x0, p0, a0]

if(1):
    T = 10.
    h = 1.00*abs(3.0*x0/p0)
    T = 20*h
    nT = int(T/h)
    #nT = 100
    #print(h, abs(3*a0/p0), abs(6*p0/x0))
    #sys.exit()
if(1):
    T = 10.
    nT = 200
    h = T/nT

t = np.linspace(0, T, nT)
# analytic solution
x_an, p_an = sub.analytic(nT, pva, h)
H_an = 0.5*(x0**2 + p0**2)

# Taylor Expansion 
x_TE, p_TE = sub.TE(nT, pva, h)
H_TE = 0.5*(x_TE**2 + p_TE**2)

# Symplectic Euler: h*D_V:h*D_T
x_SE, p_SE = sub.SE(nT, pva, h)
H_SE = 0.5*(x_SE**2 + p_SE**2)

# Velocity Verlet
# VV: 0.5h*D_V::h*D_T::0.5h*D_V
x_VV, p_VV = sub.VV(nT, pva, h)
H_VV = 0.5*(x_VV**2 + p_VV**2)


# optimized Verlet
# OV: 0.5h*D_V::h*D_T::0.5h*D_V
x_OV, p_OV = sub.OV(nT, pva, h)
H_OV = 0.5*(x_OV**2 + p_OV**2)

# mean square error
err_r1 = np.mean((x_SE - x_an)**2)
err_r2 = np.mean((x_VV - x_an)**2)

err_v1 = np.mean((p_SE - p_an)**2)
err_v2 = np.mean((p_VV - p_an)**2)

err_H1 = np.mean((H_SE - H_an)**2)
err_H2 = np.mean((H_VV - H_an)**2)


err_SE = 0.5*h*x_SE*p_SE
err_VV = h**2*(1/12.*p_VV**2 - 1/24.*x_VV**2)  


if(0):
    print(f'MSE(r) for SE = {err_r1}')
    print(f'MSE(r) for VV = {err_r2}')
    print('====')
    print(f'MSE(p) for SE = {err_v1}')
    print(f'MSE(p) for VV = {err_v2}')
    print('====')
    print(f'MSE(H) for SE = {err_H1}')
    print(f'MSE(H) for VV = {err_H2}')



#print(len(t), len(x_an), len(p_an), len(H_an))
#sys.exit()

if(1):
    plt.figure()
    plt.subplot(311)
    plt.plot(t, x_an, 'k', label = "Analytic")
    plt.plot(t, x_SE, 'r', label = "SE")
    #plt.plot(t, x_TE, 'k-o', label = "Taylor")
    plt.plot(t, x_VV, 'g', label = "VV")
    #plt.plot(t, x_OV, 'b', label = "OV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.subplot(312)
    plt.plot(t, p_an, 'k', label = "Analytic")
    plt.plot(t, p_SE, 'r', label = "SE")
    #plt.plot(t, x_TE, 'k-o', label = "Taylor")
    plt.plot(t, p_VV, 'g', label = "VV")
    #plt.plot(t, x_OV, 'b', label = "OV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("V")
    plt.subplot(313)
    plt.hlines(y = H_an, xmin=0, xmax=T, label = "Analytic")
    plt.plot(t, H_SE, 'r', label = "SE")
    plt.plot(t, H_SE+err_SE, 'r:', label = "Shadow H_SE")
    plt.plot(t, H_VV, 'g', label = "VV")
    plt.plot(t, H_VV+err_VV, 'g:', label = "Shadow H_VV")
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("H")
    plt.suptitle(f'dt = {h:.2f}')





plt.show()


