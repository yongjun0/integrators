import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys

############################
# 4th order Runge-Kutta
def rk4(dt, pva, f):
     
    x = pva[0]
    v = pva[1]
    x1 = x
    v1 = v
    y = [x1, v1]
    a1 = f(y)

    x2 = x + 0.5*v1*dt
    v2 = v + 0.5*a1*dt
    y = [x2, v2]
    a2 = f(y)

    x3 = x + 0.5*v2*dt
    v3 = v + 0.5*a2*dt
    y = [x3, v3]
    a3 = f(y)

    x4 = x + v3*dt
    v4 = v + a3*dt
    y = [x4, v4]
    a4 = f(y)

    xf = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
    vf = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)

    return xf, vf, a4

def solve(dt, Nstep, pva0, f):
    pos = np.zeros(Nstep)
    vel = np.zeros(Nstep)
    pva = pva0
    for i in range(Nstep):
        pva = rk4(dt, pva, f)
        pos[i] = pva[0]
        vel[i] = pva[1]

    return pos, vel

    
