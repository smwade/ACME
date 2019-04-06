"""Volume 2 Lab 12: Gaussian Quadrature
Sean Wade
Math 321
3/12/2015
"""

import numpy as np
from scipy import integrate
from scipy.sparse import diags
from matplotlib import pyplot as plt

def pr1():
    G = lambda x: 9*x**2 / 8. + 45*x**2 / 8. +75*x / 8
    print G(1) - G(-1)


def shift_example():
    """Plot f(x) = x**2 on [1, 4] and the corresponding function
    ((b-a)/2)*g(x) on [-1, 1].
    """
    f = lambda x: x**2
    g = lambda x: (9./4)*x**2 + (15/2.)*x + 25/4.
    x_linspace = np.linspace(1,4,1000)
    scaled_x = np.linspace(-1,1,1000)
    y = [f(x) for x in x_linspace]
    scaled_y = [g(x) for x in scaled_x]
    plt.plot(x_linspace, y)
    plt.plot(scaled_x, scaled_y)
    plt.show()



def estimate_integral(f, a, b, points, weights):
    """Estimate the value of the integral of the function 'f' over the
    domain [a, b], given the 'points' to use for sampling, and their
    corresponding 'weights'.

    Return the value of the integral.
    """
    g = lambda x: f((b - a) / 2 * x + (a +b) / 2)
    integral = (b - a)/2 * np.inner(weights, g(points))

def pr4(gamma, alpha, beta):
    a = -1 * beta / alpha
    #for i, x in enumerate(alpha):
      #  b = (gamma[i+1]/alpha[i]*alpha[i+1])**-.5
    b = np.sqrt(gamma[1:]/(alpha[:-1]*alpha[1:]))
    return np.diag(b, -1) + np.diag(a, 0) + np.diag(b, 1)

def pr5(n, a, b):
    i = np.arange(1, n+1)
    alpha = (2*i-1)/i.astype(float)
    beta = np.zeros(n)
    gamma = (i-1)/i.astype(float)
    jacobi = pr4(gamma, alpha, beta)
    eig_val, eig_vec = np.linalg.eig(jacobi)
    x = eig_val
    w = abs(b - a) * eig_vec[0]**2
    return w, x

# w,x = pr5(5,-1,1)
# print("w: %s" % w)
# print("x: %s" % x)

def pr6(f,a,b,n):
    pts, weights = pr5(n, -1, 1)
    return estimate_integral(f,a,b,pts,weights)


def gaussian_quadrature(f, a, b, n):
    """Using the functions from the previous problems, integrate the function
	'f' over the domain [a,b] using 'n' points in the quadrature.
	"""
    pts, weights = pr5(n, -1, 1)
    return estimate_integral(f,a,b,pts,weights)


def normal_cdf(x):
    """Compute the CDF of the standard normal distribution at the point 'x'.
    That is, compute P(X <= x), where X is a normally distributed random
    variable.
    """
    p = lambda t: np.exp(-t**2/2.)/np.sqrt(2*np.pi)
    print integrate.quad(p,-float('inf'), x)[0]

