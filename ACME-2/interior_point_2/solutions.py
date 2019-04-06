# Name this file 'solutions.py'.
"""Volume II: Interior Point II (Quadratic Optimization).
Sean Wade
Interior Point II
"""

import numpy as np
from scipy import linalg as la
import scipy as sp
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import cvxopt as opt

# Auxiliary function for problem 2
def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Inputs:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # Helper Functions
    #-----------------------------------------------------
    def F(x, y, mu):
        Y = np.diag(y)
        M = np.diag(mu)
        f_1 = np.dot(Q,x) - np.dot(A.T,mu) + c
        f_2 = np.dot(A,x) - y - b
        f_3 = np.dot(Y,M).dot(np.ones(m))
        return  np.hstack((f_1, f_2, f_3))
   
    # make sure y
    def calc_nu(y, mu):
        return np.dot(y, mu) / len(y)

    def search_direction(x, y, mu, nu,m,n, sigma=.1):
        DF = np.bmat([[Q, np.zeros((n,m)), -A.T], [A, -np.eye(m), np.zeros((m,m))], [np.zeros((m,n)), M, Y]])
        b = -1 *F(x, y, mu) + np.hstack((np.zeros(len(x)), np.zeros(len(y)), sigma* nu*np.ones(len(mu))))
        result = la.lu_solve(la.lu_factor(DF), b)
        return result[:n], result[n:n+m], result[n+m:]

    def step_size(delta_y, delta_mu, y, mu):
        t = .95
        mask = delta_mu < 0
        try:
            beta_max = min(1, np.amin(-1 *mu[mask]/delta_mu[mask])*t)
        except:
            beta_max = 1 

        mask = delta_y < 0
        try: 
            r =  np.amin(-1 *y[mask]/delta_y[mask])
            delta_max = min(1, r*t)
        except:
            delta_max = 1

        alpha_max = min(beta_max, delta_max)
    
        return alpha_max

    #-----------------------------------------------------
    m,n = A.shape
    x, y, mu = startingPoint(Q,c,A,b,guess)
    iter_num = 0
    nu = calc_nu(y, mu)
    while nu > tol and iter_num < niter:
        nu = calc_nu(y, mu)
        M,Y = np.diag(mu), np.diag(y)
        delta_x, delta_y, delta_mu = search_direction(x,y,mu,nu,m,n)
        alpha = step_size(delta_y, delta_mu, y, mu)
        x = x + (delta_x * alpha)
        y = y + (delta_y * alpha)
        mu = mu + (delta_mu * alpha)
        iter_num += 1

    return x, np.dot(c,x)


# Auxiliary function for problem 3
def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Plot and show the solution.
    """
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()
    # Set initial guesses.
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    H = laplacian(n)
    c = np.ones(n**2) * -1 * (n-1)**(-2)
    A = np.eye(n**2)
    # TODO Finish A

    A[2,2] = 1
    A[14,14] = 1
    A[7,7] = 1

    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))
    # Plot the solution.
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Use the data in the specified file to estimate a covariance matrix and
    expected rates of return. Find the optimal portfolio that guarantees an
    expected return of R = 1.13, with and then without short selling.

    Returns:
        An array of the percentages per asset, allowing short selling.
        An array of the percentages per asset without allowing short selling.
    """
    data = np.loadtxt('portfolio.txt')
    assets = data[:,1:]

    m, n = np.shape(assets)
    mu = np.sum(assets,axis=0)/m
    Q = opt.matrix(np.cov(assets.T))
    A = opt.matrix(np.vstack((np.ones(n), mu)))
    b = opt.matrix(np.array([1,1.13]))
    G1 = opt.matrix(np.zeros((n,n)))
    G2 = opt.matrix(-np.eye(n))
    h = opt.matrix(np.zeros(n))
    c = opt.matrix(np.zeros(n))

    sol1 = opt.solvers.qp(Q,c,G1,h,A,b)
    sol2 = opt.solvers.qp(Q,c,G2,h,A,b)
    return np.array(sol1['x']).flatten(), np.array(sol2['x']).flatten()



def test1():
    def guess(x):
        return x, np.ones(m), np.ones(m)

    Q = np.array([[1,-1],[-1,2]])
    c = np.array([-2,-6])
    A = np.array([[-1,-1],[1,-2],[-2,-1],[1,0],[0,1]])
    m,n = A.shape
    b = np.array([-2,-2,-3,0,0])
    x_guess = [.5,.5]

    print qInteriorPoint(Q,c,A,b,guess(x_guess))
