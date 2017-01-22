# Sean Wade
# 10/22/15
# lab 7

import numpy as np
import scipy as sp
from scipy import linalg as la
from matplotlib import pyplot as plt
from numpy.lib import scimath


# Problem 1
def least_squares(A,b):
    """Return the least squares solutions to Ax = b using QR decomposition."""
    m, n = A.shape
    Q, R = la.qr(A)
    qbt = np.dot(Q.T, b)
    return la.solve_triangular(R[:n],qbt[:n])
    

# Problem 2
def line_fit():
    """Plot linepts and its best-fit line on the same plot."""
    data = np.load('data.npz')
    linepts = data["linepts"]
    m,n = linepts.shape
    b = np.ones(m)
    xp = linepts[:,0]
    yp = linepts[:,1]
    mat = np.vstack((xp,b)).T
    k = la.lstsq(mat, yp)[0]
    xO = np.linspace(0,4000,100)
    yO = k[0]*xO + k[1]
    plt.xlim(0,4000)
    plt.plot(xp, yp, 'ro', xO, yO)
    plt.show()



# Problem 3
def ellipse_fit():
    """Plot ellipsepts and its best-fit line on the same plot."""
    """
    data = np.load('data.npz')
    linepts = data["ellipsepts"]
    m,n = linepts.shape
    xp = linepts[:,0]
    yp = linepts[:,1]
    A = np.vstack((2*xp, 2*yp, np.ones(m))).T
    b = xp**2 + yp**2
    c1, c2, c3 = la.lstsq(A, b)[0]
    r = np.sqrt(c1**2 + c2**2 + c3)

    theta = np.linspace(0,2*np.pi, 200)
    plt.plot(r*np.cos(theta)+c1, r*np.sin(theta)+c2, '-',xp, yp,'*')
    plt.show()
    """

    data = np.load('data.npz')
    linepts = data["ellipsepts"]
    m,n = linepts.shape
    xp = linepts[:,0]
    yp = linepts[:,1]
    A = np.vstack((xp**2, xp, xp*yp, yp, yp**2)).T
    b = np.ones(m)
    f = la.lstsq(A, b)[0]
    plot_ellipse(xp, yp, f[0], f[1], f[2], f[3], f[4])
    


def plot_ellipse(X, Y, a, b, c, d, e):
    """Plots an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1.

    Input:
      X (array) - x-coordinates of all the data points.
      Y (array) - y-coordinates of all the data points.
      a,b,c,d,e (float) - the coefficients from the equation of an 
                    ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1.
    """
    def get_r(a, b, c, d, e):
        theta = np.linspace(0,2*np.pi,200)
        A = a*(np.cos(theta)**2) + c*np.cos(theta)*np.sin(theta) + e*(np.sin(theta)**2)
        B = b*np.cos(theta) + d*np.sin(theta)
        r = (-B + np.sqrt(B**2 + 4*A))/(2*A)
        return r, theta
        
    r,theta = get_r(a,b,c,d,e)
    plt.plot(r*np.cos(theta), r*np.sin(theta), color = "r")
    plt.plot(X,Y,".", color = "b")
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

# Problem 4
def power_method(A,tol):
    """Return the dominant eigenvalue of A and its corresponding eigenvector."""
    x = np.vstack(np.random.rand(A.shape[0]))
    x = x / la.norm(x)
    nx = np.dot(A, x)
    nx = nx / la.norm(nx)
    while la.norm(x - nx) >= tol:
        x = nx
        nx = np.dot(A, x)
        nx = nx / la.norm(nx)
    print nx
    lam = np.dot(np.dot(A,nx).T, nx)/la.norm(nx)**2
    return nx, lam


    
# Problem 5
def QR_algorithm(A,niter,tol):
    """Return the eigenvalues of A using the QR algorithm."""
    A = la.hessenberg(A)
    m,n = A.shape
    eigen_vals = []
    for x in xrange(0,niter):
        Q, R = la.qr(A)
        A_new = np.dot(R, Q)
        A = A_new
    
    x = 0
    while x < m:
        if x == m-1:
            eigen_vals.append(A[x][x])
            break
        elif abs(A[x+1][x]) < tol:
            eigen_vals.append(A[x][x])
        else:
            a = A[x][x]
            b = A[x][x+1]
            c = A[x+1][x]
            d = A[x+1][x+1]
            t = a+d
            d = a*d-b*c
            eigen_vals.append(t/2.0+ scimath.sqrt(t**2/(4-d)))
            eigen_vals.append(t/2.0 - scimath.sqrt(t**2/(4-d)))
            x += 1

        x += 1

    return eigen_vals



if __name__ == '__main__':
    A = np.random.randint(0,100,(6,6))
    A = np.matrix([[4,12,17,-2],[-5.5, -30.5, -45.5, 9.5], [3., 20., 30., -6.], [1.5, 1.5, 1.5, 1.5]])
    b = np.random.randint(0,100,(6, 1))
    print QR_algorithm(A, 100000, .00001)
    print la.eigvals(A)
