# name this file solutions.py
"""Volume 2 Lab 14: Optimization Packages II (CVXOPT)
Sean Wade
MATH 323

"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix
import cvxopt as cvex
from scipy import linalg as la


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + y + 3z     >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    c = np.array([2., 1., 3.])
    G = np.array([[-1.,-2.,0.], [-2.,-1.,-3.], [-1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.]])
    h = np.array([-3.,-10.,0.,0.,0.])

    # Now convert to CVXOPT matrix type
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)

    sol = cvex.solvers.lp(c, G, h)
    return sol['x'], sol['primal objective']


def prob2():
    """Solve the transportation problem by converting all equality constraints
    into inequality constraints.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    c = matrix([4., 7., 6., 8., 8., 9])
    G = matrix([[-1.,0.,0.,0.,0.,0., 1., 0., 0., 1., 0., -1., 0., 0., -1., 0.],
                [0.,-1.,0.,0.,0.,0., 1., 0., 0., 0., 1., -1., 0., 0., 0., -1.],
                [0.,0.,-1.,0.,0.,0., 0., 1., 0., 1., 0., 0., -1., 0., -1., 0.],
                [0.,0.,0.,-1.,0.,0., 0., 1., 0., 0., 1., 0., -1., 0., 0., -1.],
                [0.,0.,0.,0.,-1.,0., 0., 0., 1., 1., 0., 0., 0., -1., -1., 0.],
                [0.,0.,0.,0.,0.,-1., 0., 0., 1., 0., 1., 0., 0., -1., 0., -1.]])
    h = matrix([0.,0.,0.,0.,0.,0.,7., 2., 4., 5., 8, -7., -2., -4., -5., -8])
    sol = cvex.solvers.lp(c, G, h)
    return sol['x'], sol['primal objective']
    

def prob3():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective'])
    """
    Q = matrix([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]])
    p = matrix([3.,0.,1.])

    sol = cvex.solvers.qp(Q,p)
    return sol['x'], sol['primal objective']


def prob4():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer (sol['x'])
        The optimal value (sol['primal objective']*-1000)
    """
    data = np.load('ForestData.npy')

    # part 1
    a = np.array([1,1,1])
    al = a * -1
    a = la.block_diag(a,a,a,a,a,a,a)
    b = la.block_diag(al,al,al,al,al,al,al)
    c = -1 * data[:,4]
    d = -1 * data[:,5]
    e = -1 * data[:,6]
    f = -1 * np.eye(21)
    G = np.vstack((a,b,c,d,e,f))

    # part 2
    h = data[::3, 1]
    i = -1. * h
    j = [-40000.]
    j2 = [-5]
    k = [-70 * 788.]
    l = np.zeros(21)
    B = np.hstack((h,i,j,j2,k,l))

    
    # part x
    m = -data[:,3]


    c = matrix(m)
    G = matrix(G)
    h = matrix(B)
    
    sol = cvex.solvers.lp(c, G, h)
    return sol['x'], sol['primal objective'] * -1000




print prob4()


