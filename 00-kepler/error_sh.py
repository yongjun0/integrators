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

import rk


x0 = 0.3
p0 = -np.sqrt(1-x0**2)
a0 = -x0
pva = [x0, p0, a0]

if(1):
    T = 20.
    h = -3.0*x0/p0
    nT = int(T/h) - 2
    #nT = 100
    #print(h, nT)
    #sys.exit()
if(1):
    T = 2.
    nT = 10
    h = T/nT

t = np.linspace(0, T, nT)


# Symplectic Euler
x_SE = np.zeros(nT)
p_SE = np.zeros(nT)

x_SE[0] = x0
p_SE[0] = p0

# VV
x_VV = np.zeros(nT)
p_VV = np.zeros(nT)

x_VV[0] = x0
p_VV[0] = p0

# analytic sol
x = np.zeros(nT)
p = np.zeros(nT)

x[0] = x0
p[0] = p0

H = 0.5*(x0**2 + p0**2)

# Taylor expansion
x_T = np.zeros(nT)
p_T = np.zeros(nT)

x_T[0] = x0
p_T[0] = p0


# Symplectic Euler: h*D_V:h*D_T
for i in range(nT-1):
    x_SE[i+1] = x_SE[i] + h*p_SE[i]     # q_{i+1} = ...DT
    p_SE[i+1] = p_SE[i] - h*x_SE[i+1]   # p_{i+1} = ...DV

# VV: 0.5h*D_V::h*D_T::0.5h*D_V
for i in range(nT-1):
    p_VV[i+1] = p_VV[i] - 0.5*h*x_VV[i]
    x_VV[i+1] = x_VV[i] + h*p_VV[i+1]
    p_VV[i+1] = p_VV[i+1] - 0.5*h*x_VV[i+1]

# analytic sol
for i in range(nT-1):
    x[i+1] = np.cos(h)*x[i] + np.sin(h)*p[i]
    p[i+1] = np.cos(h)*p[i] - np.sin(h)*x[i]

# talor expansion
for i in range(nT-1):
    x_T[i+1] = x[i] + h*p[i] - 0.5*h**2*x_T[i]
    p_T[i+1] = p_T[i] - h*x_T[i] + 0.5*h**2*p_T[i]

# symplectic Euler
H_SE = 0.5*(p_SE**2 + x_SE**2)  # not conserved!
err_SE = 0.5*h*p_SE*x_SE
# H_SE + err_SE = conserved


# VV
H_VV = 0.5*(p_VV**2 + x_VV**2)
err_VV = h**2*(1/12.*p_VV**2 - 1/24.*x_VV**2)  





if(1):
    plt.figure()
    if(0):
        plt.plot(t, x, 'b-o', label = "Analytic")
        plt.plot(t, x_SE, 'r-o', label = "SE")
        plt.plot(t, x_VV, 'g-o', label = "VV")
    if(1):
        plt.plot(t, x, 'b', label = "Analytic")
        plt.plot(t, x_SE, 'r', label = "SE")
        plt.plot(t, x_VV, 'g', label = "VV")
    if(0):
        plt.loglog(t, abs(x), 'b', label = "Analytic")
        plt.loglog(t, abs(x_SE), 'r', label = "SE")
        plt.loglog(t, abs(x_VV), 'g', label = "VV")
    #plt.plot(t, x_T, 'k', label = "Taylor")
    
    plt.legend(loc = "best")
    plt.xlabel("t")
    plt.ylabel("X")

plt.show()


