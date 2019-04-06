# Sean Wade
# Psedospectra

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problem 1
def ps_scatter_plot(A, epsilon=.001, num_pts=20):
    '''Plots the 'poorman's pseudospectrum' of a matrix A
    Parameters:
    A : ndarray of size (n,n)
    The matrix whose pseudospectrum is to be plotted.
    epsilon : float
    The norm of the random matrices that are generated.
    Defaults to 10**-3
    num_pts : int
    The number of matrices, E, that will be used in the
    algorithm. Defaults to 20.
    '''
    m, n = A.shape
    eigens = la.eig(A, right = False)

    for i in xrange(num_pts):
        E = np.random.rand(m, m)
        E = E/la.norm(E, ord=2) * epsilon
        eigens2 = la.eig((A+E), right = False)
        plt.scatter(eigens2.real, eigens2.imag, c = 'green')

    plt.scatter(eigens.real, eigens.imag, c = 'red')
    plt.show()




# Problem 2
def ps_contour_plot(A, m = 20,epsilon_vals=None):
    '''Plots the pseudospectrum of the matrix A as a contour plot.  Also,
    plots the eigenvalues.
    Parameters:
        A : square, 2D ndarray
            The matrix whose pseudospectrum is to be plotted
        m : int
            accuracy
        epsilon_vals : list of floats
            If k is in epsilon_vals, then the epsilon-pseudospectrum
            is plotted for epsilon=10**-k
            If epsilon_vals=None, the defaults of plt.contour() are used
            instead of any specified values.
    '''
    n = A.shape[0]
    T = la.schur(A)[0]
    eigsA = np.diagonal(T)
    xvals, yvals = ps_grid(eigsA, m)
    sigmin = np.zeros((m, m))
    for k in xrange(m):
        for j in xrange(m):
            T1 = (xvals[k] + 1j*yvals[j]) * np.eye(n) - T
            T2 = T1.T.conjugate()
            sigold = 0
            qold = np.zeros((n, 1))
            beta = 0
            H = np.zeros((n, n))
            q = np.random.normal(size=(n, 1)) + 1j * np.random.normal(size=(n, 1))
            q = q/la.norm(q, ord=2)
            for p in xrange(n-1):
                b1 = la.solve(T2, q)
                b2 = la.solve(T1, b1)
                v = b2 - beta * qold
                alpha = np.real(np.vdot(q,v))
                v = v - alpha * q
                beta = la.norm(v)
                qold = q
                q = v/beta
                H[p+1, p] = beta
                H[p, p+1] = beta
                H[p, p] = alpha
                sig = np.abs(np.max(la.eig(H[:p+1,:p+1])[0]))
                if np.abs(sigold/sig - 1) < .001:
                    break
                sigold = sig
            sigmin[j, k] = np.sqrt(sig)
    plt.contour(xvals,yvals,np.log10(sigmin), levels=epsilon_vals)
    plt.scatter(la.eig(A)[0].real, la.eig(A)[0].imag)
    plt.show()



def problem3(n=120,epsilon=.001,num_pts=20):
    '''
    parameters:
    n : int
    the size of the matrix to use. defaults to a 120x120 matrix.
    epsilon : float
    the norm of the random matrices that are generated.
    defaults to 10**-3
    num_pts : int
    the number of matrices, e, that will be used in the
    algorithm. defaults to 20.
    '''

    i = np.diag(np.ones(n-1) * -1j, -1)
    ones = np.diag(np.ones(n-2), -2)
    neg_i = np.diag(np.ones(n-1) * 1j, 1)
    neg_ones = np.diag(-np.ones(n-2), 2)
    A = i + ones + neg_i + neg_ones
    ps_contour_plot(A)

    i = np.diag(np.ones(n-1) * -1j, -1)
    ones = np.diag(-np.ones(n-2), -2)
    neg_i = np.diag(np.ones(n-1) * 1j, 1)
    neg_ones = np.diag(-np.ones(n-2), 2)
    A = i + ones + neg_i + neg_ones
    ps_contour_plot(A)

    ones = np.diag(np.random.rand(n) + 1j * np.random.rand(n))
    neg_i = np.diag(np.ones(n-1) * 1j, 1)
    neg_ones = np.diag(-np.ones(n-2), 2)
    A = ones + neg_i + neg_ones
    ps_contour_plot(A)


def ps_grid(eig_vals, grid_dim):
    """
        Computes the grid on which to plot the pseudospectrum
        of a matrix. This is a helper function for ps_contour_plot().
        """
    x0, x1 = min(eig_vals.real), max(eig_vals.real)
    y0, y1 = min(eig_vals.imag), max(eig_vals.imag)
    xmid = (x0 + x1) /2.
    xlen1 = x1 - x0 +.01
    ymid = (y0 + y1) / 2.
    ylen1 = y1 - y0 + .01
    xlen = max(xlen1, ylen1/2.)
    ylen = max(xlen1/2., ylen1)
    x0 = xmid - xlen
    x1 = xmid + xlen
    y0 = ymid - ylen
    y1 = ymid + ylen
    x = np.linspace(x0, x1, grid_dim)
    y = np.linspace(y0, y1, grid_dim)
    return x,y


# ====== TEST CASES ========= #
def test1():
    n = 120
    A = np.zeros((120,120))
    m2 = np.ones(n-1) * -1j
    m3 = np.ones(n-1) * 1j
    m4 = np.ones(n-2)
    m5 = np.ones(n-2) * -1
    A = A + np.diag(m4,-2) + np.diag(m2,-1) + np.diag(m3,1) + np.diag(m5,2)
    ps_scatter_plot(A)

def test2():
    n = 120
    A = np.zeros((120,120))
    m2 = np.ones(n-1) * -1j
    m3 = np.ones(n-1) * 1j
    m4 = np.ones(n-2)
    m5 = np.ones(n-2) * -1
    A = A + np.diag(m4,-2) + np.diag(m2,-1) + np.diag(m3,1) + np.diag(m5,2)
    ps_contour_plot(A)

def test3():
    problem3()

