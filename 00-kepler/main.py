''' Test Kepler 2 body problem
[1]	X. Tu, A. Murua, and Y. Tang, “New high order symplectic integrators via generating functions with its application in many-body problem,” BIT Numerical Mathematics, pp. 1–27, Nov. 2019.
see sec. 4.1 on [1]

also, 
https://sites.google.com/a/ucsc.edu/krumholz/teaching-and-courses/ast119_w15/class-11
https://en.wikipedia.org/wiki/Kepler%27s_equation
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
e = 0.6
r0 = 1-e
v0= np.sqrt((1+e)/(1-e))

a = 1./(2/r0 - v0**2)
b = a*np.sqrt(1 - e**2)
tau = 2*np.pi*a**(1.5)

q1_0 = r0
q2_0 = 0
p1_0 = 0
p2_0 = v0

tmax = 1.1*tau # tau = 2pi
nbin = 10
dt_start = 1.0e-3*tau
dt_end = 1.0e-2*tau
dt_start = 1.0e-3
dt_end = 3.0e-1

print(q1_0, p2_0)
print(dt_start)
print(dt_end)
print(tmax)
sys.exit()


log_dt_start = np.log10(dt_start)
log_dt_end = np.log10(dt_end)
log_dt_step = (log_dt_end - log_dt_start)/(nbin-1)

dt = np.zeros(nbin)
Nstep = np.zeros(nbin, dtype = int)
err_ns = np.zeros(nbin)

for i in range(nbin):
    indx = log_dt_start + log_dt_step*(i)
    dt[i] = 10**(indx)
#    dt[i] = (i+1)*(tmax/1000.)
#    dt[i] = dt_start + (dt_end-dt_start)/nbin*(i)
    Nstep[i] = int(tmax/dt[i])
    
    if(Nstep[i] <= 1):
        print("Nstep is too small.", i, Nstep[i])
        sys.exit()

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


# exact solution
for i in range(nbin):
    t0 = np.arange(0, tmax, dt[i])
    pos, vel = kepler2bdy.solve(t0, r0, v0, a, b, e)
    pos1 = pos[0]
    pos2 = pos[1]
    vel1 = vel[0]
    vel2 = vel[1]

    #H0[i] = 0.5*(vel1[-1]**2 + vel2[-1]**2) - 1/np.sqrt(pos1[-1]**2 + pos2[-1]**2) # Hamiltonian
    H0 = 0.5*(vel1[-1]**2 + vel2[-1]**2) - 1/np.sqrt(pos1[-1]**2 + pos2[-1]**2) # Hamiltonian



# symplectic integrator solution
    pva1 = [q1_0, p1_0, acc1_0]
    pva2 = [q2_0, p2_0, acc2_0]

    pos_ns, vel_ns  = symplectic.solve(dt[i], Nstep[i], pva1, pva2, kepler2bdy.acc, \
            order = order1)

    pos1_ns = pos_ns[0]
    pos2_ns = pos_ns[1]
    vel1_ns = vel_ns[0]
    vel2_ns = vel_ns[1]

    pos1_ns = np.insert(pos1_ns, 0, pos1[0])
    vel1_ns = np.insert(vel1_ns, 0, vel1[0])
    pos2_ns = np.insert(pos2_ns, 0, pos2[0])
    vel2_ns = np.insert(vel2_ns, 0, vel2[0])

    #H_ns = 0.5*(vel1_ns[-1]**2 + vel2_ns[-1]**2) - 1/np.sqrt(pos1_ns[-1]**2 + pos2_ns[-1]**2) 
    #err_ns[i] = abs((H0 - H_ns)/H0)
    H_ns = 0.5*(vel1_ns**2 + vel2_ns**2) - 1/np.sqrt(pos1_ns**2 + pos2_ns**2) 
    err_ns[i] = np.std(H_ns)

    #print('err: ', err_ns[i])

if(1):
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
    fig_name = fig_folder+"/"+"order_"+str(order1)+".png"
    fss = 15
    fl = 18
    plt.figure()
    plt.loglog(dt, err_ns, 'b.', label = "Symplectic, order = "+str(ord1))
#plt.loglog(dt, err_20d, 'g', label = "Verlet")
#plt.loglog(dt, dt, linewidth=3, label = "power = 1", alpha=0.5)
    #plt.loglog(dt, pow4, linewidth=3, label = "power = 4", alpha=0.5)
    plt.loglog(dt, powN, linewidth=3, label = "power = "+str(ord1), alpha=0.5)
    #plt.loglog(dt, pow7, linewidth=3, label = "power = 7", alpha=0.5)

    plt.legend(loc = "best", fontsize = fss) 
    plt.xlabel("dt", fontsize = fss)
    plt.ylabel("Error(H)", fontsize = fss)
#plt.xticks(fontsize = fl)
#plt.yticks(fontsize = fl)
plt.show()
#plt.savefig(fig_name)


