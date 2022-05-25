import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys

# analytic sol
def analytic(nT, pva, h):
    x = np.zeros(nT)
    p = np.zeros(nT)
    x0, p0, a0  = pva
    x[0] = x0
    p[0] = p0

    for i in range(nT-1):
        x[i+1] = np.cos(h)*x[i] + np.sin(h)*p[i]
        p[i+1] = np.cos(h)*p[i] - np.sin(h)*x[i]
    return x, p

# symplectic Euler
def SE(nT, pva, h):
    x = np.zeros(nT)
    p = np.zeros(nT)
    x0, p0, a0  = pva
    x[0] = x0
    p[0] = p0

    for i in range(nT-1):
        x[i+1] = x[i] + h*p[i]     # q_{i+1} = ...DT
        p[i+1] = p[i] - h*x[i+1]   # p_{i+1} = ...DV

    return x, p

# velocity verlet
def VV(nT, pva, h):
# VV: 0.5h*D_V::h*D_T::0.5h*D_V
    x = np.zeros(nT)
    p = np.zeros(nT)
    x0, p0, a0  = pva
    x[0] = x0
    p[0] = p0

    for i in range(nT-1):
        p[i+1] = p[i] - 0.5*h*x[i]
        x[i+1] = x[i] + h*p[i+1]
        p[i+1] = p[i+1] - 0.5*h*x[i+1]
        
        #x[i+1] = x[i+1] + 1/6.*h**3*(p[i])
        #p[i+1] = p[i+1] - 1/12.*h**3*(x[i])


    return x, p

# Taylor expansion
def TE(nT, pva, h):
    x = np.zeros(nT)
    p = np.zeros(nT)
    x0, p0, a0  = pva
    x[0] = x0
    p[0] = p0
    # talor expansion
    for i in range(nT-1):
        x[i+1] = x[i] + h*p[i] + 0.5*h**2*(-x[i]) #+ 1/6.*h**3*(-p[i]) + 1/24.*h**4*(x[i])
        p[i+1] = p[i] + h*(-x[i]) + 0.5*h**2*(-p[i]) #+ 1/6.*h**3*x[i] + 1/24.*h**4*p[i]

    return x, p

# optimized verlet
def OV(nT, pva, h):
    x = np.zeros(nT)
    p = np.zeros(nT)
    x0, p0, a0  = pva
    x[0] = x0
    p[0] = p0
    a = a0

    xx = 2*np.sqrt(326)-36
    yy = np.power(xx, 1./3.)
    zz = (yy**2 + 6*yy - 2)/(12*yy)
    
    c = [zz, 1-2*zz, zz]
    d = [0.5, 0.5, 0]
    n = len(c)
    '''
    for i in range(0, n-1):
        v = v + h*c[i]*a
        x = x + h*d[i]*v

        a = f(mass, x)

    v = v + h*c[-1]*a
    '''
    
    for i in range(nT-1):
        p[i+1] = p[i] + h*c[0]*a
        x[i+1] = x[i] + h*d[0]*p[i+1]
        a = -x[i+1]     # force calc
        #---
        p[i+1] = p[i+1] + h*c[1]*a
        x[i+1] = x[i+1] + h*d[1]*p[i+1]
        a = -x[i+1]     # force calc
        #---
        p[i+1] = p[i+1] + h*c[2]*a

    return x, p
