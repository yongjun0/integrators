'''
Solve 2body Kepler system

[1]	X. Tu, A. Murua, and Y. Tang, “New high order symplectic integrators via generating functions with its application in many-body problem,” BIT Numerical Mathematics, pp. 1–27, Nov. 2019.
see sec. 4.1 on [1]

also, 
https://sites.google.com/a/ucsc.edu/krumholz/teaching-and-courses/ast119_w15/class-11
https://en.wikipedia.org/wiki/Kepler%27s_equation
'''

import numpy as np
import numba as nb
import scipy.special as special

def acc(r, v):
    q1 = r[0]
    q2 = r[1]
    a1 = -q1/(q1**2 + q2**2)**(1.5)
    a2 = a1/q1*q2
    a = np.array([a1, a2])
    return a


def acc(q1, q2, p1, p2):
    a1 = -q1/(q1**2 + q2**2)**(1.5)
    a2 = a1/q1*q2
    return a1, a2


#@nb.njit
def solve(t, r0, v0, a, b, e):

    Nstep = len(t)
    dt = t[1]-t[0]

    pos1 = np.zeros((2, Nstep))
    vel1 = np.zeros((2, Nstep))
    
    guess = 0
    for i in range(Nstep):
        pos1[:, i], vel1[:, i], guess = solution(a, b, e, guess, t[i])

    return pos1, vel1

#@nb.njit
############################
def solution(a, b, e, guess, t):
    c1 = a**(-1.5)*t
    f = lambda x: x - e*np.sin(x) - c1
    Df = lambda x: 1 - e*np.cos(x)
    E = newton(f, Df, guess, 1e-8, 10)

    q1 = a*(np.cos(E) - e)
    q2 = b*np.sin(E)
    dEdt = a**(-1.5)/(1 - e*np.cos(E))
    p1 = -a*np.sin(E)*dEdt
    p2 = b*np.cos(E)*dEdt

    r = np.array([q1, q2])
    v = np.array([p1, p2])

    return r, v, E





def newton(f, Df, x0, epsilon,max_iter):
    '''
    https://www.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/
    Approximate solution of f(x)=0 by Newton's method.
    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            #print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None



