"""Volume 2 Lab 19: Interior Point 1 (Linear Programs)
Sean Wade
Volume II
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt

# Auxiliary Functions ---------------------------------------------------------
def startingPoint(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A.dot(A.T))
    x = A.T.dot(B.dot(b))
    lam = B.dot(A.dot(c))
    mu = c - A.T.dot(lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(m):
    """Generate a 'square' linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add slack variables.
    Inputs:
        m -- positive integer: the number of desired constraints
             and the dimension of space in which to optimize.
    Outputs:
        A -- array of shape (m,n).
        b -- array of shape (m,).
        c -- array of shape (n,).
        x -- the solution to the LP.
    """
    n = m
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(n)*10
    b = A.dot(x)
    c = A.sum(axis=0)/float(n)
    return A, b, -c, x

# This random linear program generator is more general than the first.
def randomLP2(m,n):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Inputs:
        m -- positive integer >= n, number of desired constraints
        n -- dimension of space in which to optimize
    Outputs:
        A -- array of shape (m,n+m)
        b -- array of shape (m,)
        c -- array of shape (n+m,), with m trailing 0s
        v -- the solution to the LP
    """
    A = np.random.random((m,n))*20 - 10
    A[A[:,-1]<0] *= -1
    v = np.random.random(n)*10
    k = n
    b = np.zeros(m)
    b[:k] = A[:k,:].dot(v)
    b[k:] = A[k:,:].dot(v) + np.random.random(m-k)*10
    c = np.zeros(n+m)
    c[:n] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(m)))
    return A, b, -c, v


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    def F(x, lam, mu):
        f_1 = np.dot(A.T, lam) + mu - c
        f_2 = np.dot(A, x) - b
        f_3 = np.dot(np.diag(mu), x)
        return np.hstack((f_1, f_2, f_3))

    def calc_nu(x, mu):
        return np.dot(x, mu) / len(x)

    def search_direction(x, lam, mu, nu, sigma=.1):
        m,n  = A.shape
        DF = np.bmat([[np.zeros((n,n)), A.T, np.eye(n)],[A, np.zeros((m,m)), np.zeros((m,n))], [np.diag(mu), np.zeros((n,m)), np.diag(x)]])
        b = -1 *F(x, lam, mu) + np.hstack((np.zeros(len(x)), np.zeros(len(lam)), sigma* nu*np.ones(len(mu))))
        result = la.lu_solve(la.lu_factor(DF), b)
        return result[:n], result[n:n+m], result[n+m:]

    def step_size(delta_x, delta_mu, x, mu):
        n,m = A.shape
        mask = delta_mu < 0
        try:
            alpha_max = min(1, np.amin(-1 *mu[mask]/delta_mu[mask]))
        except:
            alpha_max = 1

        mask = delta_x < 0
        try:
            r =  np.amin(-1 *x[mask]/delta_x[mask])
            delta_max = min(1, r)
        except:
            delta_max = 1

        return alpha_max, delta_max



    n,m = A.shape
    x, lam, mu = startingPoint(A, b, c)
    iter_num = 0
    nu = calc_nu(x, mu)
    while nu >= tol and iter_num < niter:
        delta_x, delta_lam, delta_mu = search_direction(x, lam, mu, nu)
        alpha, delta = step_size(delta_x, delta_mu, x, mu)
        x = x + delta_x * delta
        lam = lam + delta_lam * alpha
        mu = mu + delta_mu * alpha
        nu = calc_nu(x, mu)
        iter_num += 1

    return x[:n], np.dot(c,x)


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""

    data = np.loadtxt("simdata.txt")
    domain = np.linspace(0,10,200)
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n+1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:,0]
    y[1::2] = data[:,0]
    x = data[:,1:]

    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    plt.scatter(data[:,1], data[:,0])
    plt.plot(domain, domain*beta + b)


    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    plt.title("Least Squares")
    plt.scatter(data[:,1], data[:,0])
    plt.plot(domain, domain*slope + intercept)
    plt.show()

if __name__ == '__main__':
    A, b, c, x = randomLP2(6,4)
    point, value = interiorPoint(A, b, c)
    print x
    print point[:4]
    print np.allclose(x, point[:4])
    #leastAbsoluteDeviations()
