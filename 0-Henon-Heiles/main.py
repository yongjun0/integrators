'''
Henon-Heiles
https://github.com/williamgilpin/rk4/blob/master/rk4_demo.py
https://mathworld.wolfram.com/Henon-HeilesEquation.html
'''
import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys
import time
import os

import kepler2bdy
import symplectic
import rk

# Main routine
# order1 = 20, 21, 40, 41 .... 
order1 = int(sys.argv[1])
ord1 = int(order1/10)

# initial conditions
if(1):
    q1_0 = 0.1
    q2_0 = 0.1
    p1_0 = 0.0
    p2_0 = 0.0

if(0):
    q1_0 = 0.0
    q2_0 = 0.5
    p1_0 = 0.0
    p2_0 = 0.0

if(0):
    q1_0 = 0.408248
    q2_0 = 0.0
    p1_0 = 0.0
    p2_0 = 0.0




H0 = 0.5*(p1_0**2 + p2_0**2 + q1_0**2 + q2_0**2) + q1_0**2*q2_0 -(1./3.)*q2_0**3

tmax = 100 # tau = 2pi
nbin = 10

dt_start = 0.01
dt_end = 1

log_dt_start = np.log10(dt_start)
log_dt_end = np.log10(dt_end)
log_dt_step = (log_dt_end - log_dt_start)/(nbin-1)

dt = np.zeros(nbin)
Nstep = np.zeros(nbin, dtype = int)
err_ns = np.zeros(nbin)

for i in range(nbin):
    indx = log_dt_start + log_dt_step*i
    dt[i] = 10**(indx)
    #dt[i] = dt_start + (dt_end-dt_start)/nbin*(i)
    Nstep[i] = int(tmax/dt[i])
    
Nstep = Nstep.astype(int)
err = np.zeros(nbin)
#print(dt)
#sys.exit()
# init. acc
acc1_0, acc2_0 = kepler2bdy.acc(q1_0, q2_0, p1_0, p2_0)
pva1 = [q1_0, p1_0, acc1_0]
pva2 = [q2_0, p2_0, acc2_0]

# solve 
pow4 = dt**4
pow7 = dt**7
powN = dt**ord1


# symplectic integrator solution
for i in range(nbin):
#for i in range(1):
    print(' = ', i)
    pva1 = [q1_0, p1_0, acc1_0]
    pva2 = [q2_0, p2_0, acc2_0]

    pos_ns, vel_ns  = symplectic.solve(dt[i], Nstep[i], pva1, pva2, kepler2bdy.acc, \
            order = order1)

    pos1_ns = pos_ns[0]
    pos2_ns = pos_ns[1]
    vel1_ns = vel_ns[0]
    vel2_ns = vel_ns[1]

    pos1_ns = np.insert(pos1_ns, 0, q1_0)
    vel1_ns = np.insert(vel1_ns, 0, p1_0)
    pos2_ns = np.insert(pos2_ns, 0, q2_0)
    vel2_ns = np.insert(vel2_ns, 0, p2_0)

    #H_ns = 0.5*(vel1_ns[-1]**2 + vel2_ns[-1]**2) - 1/np.sqrt(pos1_ns[-1]**2 + pos2_ns[-1]**2) 
    #err_ns[i] = abs((H0 - H_ns)/H0)
    H_ns = 0.5*(vel1_ns**2 + vel2_ns**2 + pos1_ns**2 + pos2_ns**2)  \
      + pos1_ns**2*pos2_ns -(1./3.)*pos2_ns**3
    err_ns[i] = np.std(H_ns)

    #print("pos1: ", pos1_ns)
    #print("pos2: ", pos2_ns)
    #print("vel1: ", vel1_ns)
    #print("vel2: ", vel2_ns)

    #print('err: ', err_ns[i])

    #plt.figure()
    #plt.plot(pos1_ns, vel1_ns)
    #plt.plot(pos2_ns, vel2_ns)

if(0):
    #print('err_ns  = ', err_ns)
    d_folder = "data"
    if not (os.path.exists(d_folder)):
        os.mkdir(d_folder)

    fig_folder = "figures"
    if not (os.path.exists(fig_folder)):
        os.mkdir(fig_folder)

    file_name = d_folder+"/"+"err_"+str(order1)
    np.savez(file_name, dt= dt, err = err_ns)

if(1):
    #fig_name = fig_folder+"/"+"order_"+str(order1)+".png"
    fss = 15
    fl = 18
    plt.figure()
    plt.loglog(dt, err_ns, 'b.', label = "Symplectic, order = "+str(ord1))
#plt.loglog(dt, err_20d, 'g', label = "Verlet")
#plt.loglog(dt, dt, linewidth=3, label = "power = 1", alpha=0.5)
    plt.loglog(dt, err_ns[0]/pow4[0]*pow4, linewidth=3, label = "power = 4", alpha=0.5)
    plt.loglog(dt, powN, linewidth=3, label = "power = "+str(ord1), alpha=0.5)
    #plt.loglog(dt, pow7, linewidth=3, label = "power = 7", alpha=0.5)

    plt.legend(loc = "best", fontsize = fss) 
    plt.xlabel("dt", fontsize = fss)
    plt.ylabel("Error(H)", fontsize = fss)
#plt.xticks(fontsize = fl)
#plt.yticks(fontsize = fl)
plt.show()
#plt.savefig(fig_name)


