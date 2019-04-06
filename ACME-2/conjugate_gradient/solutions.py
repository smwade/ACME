# Name this file 'solutions.py'.
"""Volume II Lab 18: Conjugate Gradient
Sean Wade
Volume 2
2/27/2016
"""

from __future__ import division
import numpy as np
from scipy import linalg as la
from scipy.optimize import fmin_cg

# Problem 1
def conjugateGradient(b, x0, Q, tol=1e-4):
    """Use thef Conjugate Gradient Method to find the solution to the linear
    system Qx = b.
    
    Parameters:
        b  ((n, ) ndarray)
        x0 ((n, ) ndarray): An initial guess for x.
        Q  ((n,n) ndarray): A positive-definite square matrix.
        tol (float)
    
    Returns:
        x ((n, ) ndarray): The solution to the linear systm Qx = b, according
            to the Conjugate Gradient Method.
    """
    r_0 = np.dot(Q, x0) - b
    d_0 = -r_0
    r_k = r_0
    d_k = d_0
    x_k = x0
    while la.norm(r_k) > tol:
        a_k = np.dot(r_k, r_k) / np.dot(d_k, np.dot(Q, d_k))
        x_k1 = x_k + np.dot(a_k, d_k)
        r_k1 = r_k + np.dot(a_k, np.dot(Q, d_k))
        b_k1 = np.dot(r_k1, r_k1) / np.dot(r_k, r_k)
        d_k1 =  -r_k1 + np.dot(b_k1, d_k)
        x_k = x_k1
        r_k = r_k1
        d_k = d_k1
    return x_k1

# Problem 2
def prob2(filename='linregression.txt'):
    """Use conjugateGradient() to solve the linear regression problem with
    the data from linregression.txt.
    Return the solution x*.
    """
    data = np.loadtxt(filename)
    A = np.copy(data)
    b = data[:,0]
    A[:,0] = 1
    Q = np.dot(A.T, A)
    b = np.dot(A.T, b)
    x0 = np.random.random(b.size)
    x = conjugateGradient(b, x0, Q)
    return x


# Problem 3
def prob3(filename='logregression.txt'):
    """Use scipy.optimize.fmin_cg() to find the maximum likelihood estimate
    for the data in logregression.txt.
    """
    data = np.loadtxt(filename)
    A = np.copy(data)
    y = data[:,0]
    A[:,0] = 1
    guess = np.array([1.,1.,1.,1.])

    def objective(b):
        #Return -1*l(b[0], b[1]), where l is the log likelihood.
        return (np.log(1+np.exp(A.dot(b))) - y*(A.dot(b))).sum()

    return fmin_cg(objective, guess)
