'''
1. Ruth, IEE Trans. Nuc. Sci. Vol. NS-30, No. 4, Aug, 1983 Check.
2. Yoshida, Physics letters A, Vol 150, no 5, 6, 7, 12 Nov. 1990
3. Neri, "Lie algebras and canonical integration", 1988
4. McLachlan and Atela, Nonlinearity 5, pp. 541-562, 1992
5. McLachlan, SIAM J. SCI. COMPUT. Vol. 16, No. 1, pp.151-168, Jan. 1995
6. M. Suzuki, Physica A 205 (1994) 65–79
7. W. Kahan & R.-C. Li, Math. Comput. 66 (1997) 1089–1099
8. M. Sofroniou & G. Spaletta, J. of Optimization Methods and Software, 
    Vol. 20, Nos, 4-5, 2005, 597-613 
'''

import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import sys
import time
import numba as nb
import symplectic_params as sym

#@nb.jit
def solve(dt, Nstep, pva1, pva2, f, order):
    '''
    See Ref. 5 for more info. 
    '''
    c, d = sym.get_cd(order)

    pos = np.zeros((2, Nstep))
    vel = np.zeros((2, Nstep))

    for i in range(Nstep):
        pva1, pva2 = symplectic(dt, pva1, pva2, f, c, d) 
        pos[0, i] = pva1[0]
        pos[1, i] = pva2[0]
        vel[0, i] = pva1[1]
        vel[1, i] = pva2[1]
    return pos, vel

############################
# symplectic integrators
#@nb.jit
def symplectic(dt, pva1, pva2, f, c, d):
    x1 = pva1[0]
    v1 = pva1[1]
    a1 = pva1[2]

    x2 = pva2[0]
    v2 = pva2[1]
    a2 = pva2[2]

    n = len(c)
    h = dt
    if(n == 1):
        v1 = v1 + h*c[0]*a1
        x1 = x1 + h*d[0]*v1

        v2 = v2 + h*c[0]*a2
        x2 = x2 + h*d[0]*v2

        a1, a2 = f(x1, x2, v1, v2)

    else:
        for i in range(0, n-1):
            v1 = v1 + h*c[i]*a1
            x1 = x1 + h*d[i]*v1

            v2 = v2 + h*c[i]*a2
            x2 = x2 + h*d[i]*v2

            a1, a2 = f(x1, x2, v1, v2)

        v1 = v1 + h*c[-1]*a1
        v2 = v2 + h*c[-1]*a2

    pva1 = [x1, v1, a1]
    pva2 = [x2, v2, a2]
    return pva1, pva2
#x1, x2, v1, v2, a1, a2
