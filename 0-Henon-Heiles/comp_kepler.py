## not working yet....
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

# m1, r1, v1: ptcl 1
# m2, r2, v2: ptcl 2
# m = m1 + m2 = 1
m1 = 0.5
m2 = 1.0 - m1
r1 = np.array([0.0, 0.0])
r2 = np.array([r0, 0.0])
v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, v0])

# convert to Kepler problem
r0 = r2 - r1
v0 = v2 - v1

tmax = 10.1*tau # tau = 2pi
nbin = 10

dt_start = 1.0e-3*tau
dt_end = 1.e-2*tau

log_dt_start = np.log10(dt_start)
log_dt_end = np.log10(dt_end)
log_dt_step = (log_dt_end - log_dt_start)/(nbin-1)

dt = np.zeros(nbin)
Nstep = np.zeros(nbin, dtype = int)
err_ns = np.zeros(nbin)

for i in range(nbin):
    indx = log_dt_start + log_dt_step*(i)
    dt[i] = 10**(indx)
    Nstep[i] = int(tmax/dt[i])
    
    if(Nstep[i] <= 1):
        print("Nstep is too small.", i, Nstep[i])
        sys.exit()

Nstep = Nstep.astype(int)
err = np.zeros(nbin)

# init. acc
acc0 = kepler2bdy.acc(r0, v0)
pva1 = [r0[0], v0[0], acc0[0]]
pva2 = [r0[1], v0[1], acc0[1]]

# solve 
pow4 = dt**4
pow7 = dt**7
powN = dt**ord1

nbin = 1
# r_ext, v_ext: exact solution
for i in range(nbin):
    t0 = np.arange(0, tmax, dt[i])
    r_ext, v_ext = kepler2bdy.solve(t0, r0, v0, a, b, e)
    H0 = 0.5*(v_ext[0]**2 + v_ext[1]**2) - 1/np.sqrt(r_ext[0]**2 + r_ext[1]**2) # Hamiltonian
    
# r_kpl, r_kpl: symplectic integrator solution of Kepler problem
for i in range(nbin):
    t0 = np.arange(0, tmax, dt[i])
    r_kpl, v_kpl  = symplectic.solve(dt[i], Nstep[i], pva1, pva2, kepler2bdy.acc_kpl,  order = order1)

    r_kpl[0] = np.insert(r_kpl[0], 0, r0[0])
    r_kpl[1] = np.insert(r_kpl[1], 0, r0[1])
    v_kpl[0] = np.insert(v_kpl[0], 0, v0[0])
    v_kpl[1] = np.insert(v_kpl[1], 0, v0[1])

    #pos1_ns = np.insert(pos1_ns, 0, pos1[0])
    #vel1_ns = np.insert(vel1_ns, 0, vel1[0])
    #pos2_ns = np.insert(pos2_ns, 0, pos2[0])
    #vel2_ns = np.insert(vel2_ns, 0, vel2[0])

    H_kpl = 0.5*(v_kpl[-1]**2 + v_kpl[-1]**2) - 1/np.sqrt(r_kpl[-1]**2 + r_kpl[-1]**2) 
    print(H_kpl)
sys.exit()
    #err_ns[i] = abs((H0 - H_ns)/H0)
    #print(f'H = {H0:.2f}, {H_ns:.2f}')
    #H_test = 0.5*(vel1_ns**2 + vel2_ns**2) - 1/np.sqrt(pos1_ns**2 + pos2_ns**2) 
    #print(f'H = {H_test}')
    #print(pos1_ns)
    #print(dt[i], Nstep[i], err_ns[i])
    #print(f'q1 = {pos1}')
    #print(f'q2 = {pos2}')
    #print(f'p1 = {vel1}')
    #print(f'p2 = {vel2}')


sys.exit()


print('err_ns  = ', err_ns)
d_folder = "data"
if not (os.path.exists(d_folder)):
    os.mkdir(d_folder)

fig_folder = "figures"
if not (os.path.exists(fig_folder)):
    os.mkdir(fig_folder)

file_name = d_folder+"/"+"err_"+str(order1)
np.savez(file_name, dt= dt, err = err_ns)

if(0):
    fig_name = fig_folder+"/"+"order_"+str(order1)+".png"
    fss = 15
    fl = 18
    plt.figure()
    plt.loglog(dt, err_ns, 'b', label = "Symplectic, order = "+str(ord1))
#plt.loglog(dt, err_20d, 'g', label = "Verlet")
#plt.loglog(dt, dt, linewidth=3, label = "power = 1", alpha=0.5)
    plt.loglog(dt, pow4, linewidth=3, label = "power = 4", alpha=0.5)
    plt.loglog(dt, powN, linewidth=3, label = "power = "+str(ord1), alpha=0.5)
    plt.loglog(dt, pow7, linewidth=3, label = "power = 7", alpha=0.5)

    plt.legend(loc = "best", fontsize = fss) 
    plt.xlabel("dt", fontsize = fss)
    plt.ylabel("Error(H)", fontsize = fss)
#plt.xticks(fontsize = fl)
#plt.yticks(fontsize = fl)
plt.show()
#    plt.savefig(fig_name)


