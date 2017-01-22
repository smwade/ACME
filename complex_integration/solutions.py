import numpy as np
from numpy import poly1d
from sympy import mpmath as mp
from scipy.integrate import quad
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def singular_surface_plot(f, x_bounds=(-1.,1), y_bounds=(-1.,1.), res=500, threshold=2., lip=.1):
    """ Plots the absolute value of a function as a surface plot """
    x = np.linspace(x_bounds[0], x_bounds[1], res)
    y = np.linspace(y_bounds[0], y_bounds[1], res)
    X, Y = np.meshgrid(x, y , copy=False)
    Z = X + 1.0j*Y
    fZ = np.abs(f(Z))

    fZ[(threshold + lip > fZ) & (fZ > threshold)] = threshold
    fZ[(-threshold - lip < fZ) & (fZ < -threshold)] = -threshold
    fZ[np.absolute(fZ) >= threshold + lip] = np.nan

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, fZ, cmap="coolwarm")
    plt.show()

def partial_fractions(p, q):
    """ Finds the partial fraction representation of the rational
        function 'p' / 'q' where 'q' is assumed to not have any repeated
        roots. 'p' and 'q' are both assumed to be numpy poly1d objects.
        Returns two arrays. One containing the coefficients for
        each term in the partial fraction expansion, and another containing
        the corresponding roots of the denominators of each term. """
    return p(q.roots)/q.deriv()(q.roots), q.roots


def cpv(p, q, tol = 1E-8):
    """ Evaluates the cauchy principal value of the integral over the
        real numbers of 'p' / 'q'. 'p' and 'q' are both assumed to be numpy
        poly1d objects. 'q' is expected to have a degree that is
        at least two higher than the degree of 'p'. Roots of 'q' with
        imaginary part of magnitude less than 'tol' are treated as if they
        had an imaginary part of 0. """
    return np.real(2 * np.pi * 1.0j * np.sum(p(q.roots[q.roots.imag > tol]) / q.deriv()(q.roots[q.roots.imag > tol])))


def count_roots(p):
    """ Counts the number of roots of the polynomial object 'p' on the
        interior of the unit ball using an integral. """
    dp = p.deriv()
    e = lambda x: np.exp(1.0j*x)
    function = lambda t: dp(e(t))*1.0j*np.exp(1.0j*t)/p(e(t))
    return quad(function, 0., 2*np.pi)[0]/(2.*np.pi*1.0j)
