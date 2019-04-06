# Sean Wade
# GMRES

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

def gmres(A, b, x0, k=100, tol=1e-8):
    '''Calculate approximate solution of Ax=b using GMRES algorithm.
        
    INPUTS:
    A    - Callable function that calculates Ax for any input vector x.
    b    - A NumPy array of length m.
    x0   - An arbitrary initial guess.
    k    - Maximum number of iterations of the GMRES algorithm. Defaults to 100.
    tol  - Stop iterating if the residual is less than 'tol'. Defaults to 1e-8.
    
    RETURN:
    Return (y, res) where 'y' is an approximate solution to Ax=b and 'res'
    is the residual.
    
    Examples:
    >>> a = np.array([[1,0,0],[0,2,0],[0,0,3]])
    >>> A = lambda x: a.dot(x)
    >>> b = np.array([1, 4, 6])
    >>> x0 = np.zeros(b.size)
    >>> gmres(A, b, x0)
    (array([ 1.,  2.,  2.]), 1.09808907533e-16)
    '''
    Q = np.empty((len(b), k+1))
    H = np.zeros((k+1,k))
    r0 = b - A(x0)
    e1 = np.zeros(k+1)
    e1[0] = 1
    beta = la.norm(b - A(x0))

    Q[:,0] = r0 / la.norm(r0)
    for j in xrange(k):
        Q[:,j+1] = A(Q[:,j])
        for i in xrange(j+1):
		    H[i,j] = np.vdot(Q[:,i].conjugate(), Q[:,j+1])
		    Q[:,j+1] = Q[:,j+1]- H[i,j]*Q[:,i]
        
        H[j+1, j] = np.sqrt(np.dot(np.conjugate(Q[:,j+1]), Q[:,j+1]))
        Q[:,j+1] = Q[:,j+1] / H[j+1,j]
        yn, res, pl, pl2 = la.lstsq(H[:j+2,:j+1], beta*e1[:j+2])
        res = np.sqrt(res)

        if res < tol:
            return np.dot(Q[:,:j+1], yn[:j+1]) + x0, res
    return np.dot(Q[:,:k], yn) + x0, res


def plot_gmres(A, b, x0, tol=1e-8):
    """Use the GMRES algorithm to approximate the solution to Ax=b. Plot
    the eigenvalues of A and the convergence of the algorithm.
    INPUTS:
    A - A 2-D NumPy array of shape mxm.
    b - A 1-D NumPy array of length m.
    x0 - An arbitrary initial guess.
    tol - Stop iterating and create the desired plots when the residual is
    less than 'tol'. Defaults to 1e-8.
    OUTPUT:
    Follow the GMRES algorithm until the residual is less than tol, for a
    maximum of m iterations. Then create the two following plots (subplots
    of a single figure):
    1. Plot the eigenvalues of A in the complex plane.
    2. Plot the convergence of the GMRES algorithm by plotting the
    iteration number on the x-axis and the residual on the y-axis.
    Use a log scale on the y-axis.
    '''
    '''
    eig_val = la.eigvals(A)
    plt.subplot(1,2,1)
    plt.scatter(eig_val.real,eig_val.imag)
    plt.title('Eig vals')
    
    x_values = []
    y_values = []
    for i in range(1,len(b)+1):
        res = gmres(A.dot,b,x0,i+1)[1]
        if res>tol:
            y_values.append(res)
            x_values.append(i)
        else:
            break
    plt.subplot(1,2,2)
    plt.plot(x_values,y_values)
    plt.title('gmres convergence')
    plt.show()
    """

    k = len(b)
    Q = np.empty((b.size,k+1))
    H = np.zeros((k+1,k))
    r0 = b - A.dot(x0)
    beta = la.norm(r0)
    e = np.zeros(k+1)
    e[0] = beta
    Q[:,0] = r0/la.norm(r0)
    toplot = []
    res = tol+1
    j = 0

    while j<k and res>tol:
        Q[:,j+1] = A.dot(Q[:,j])
        for i in xrange(j+1):
            H[i,j] = np.dot(Q[:,i].T,Q[:,j+1])
            Q[:,j+1] = Q[:,j+1]-H[i,j]*Q[:,i]
            
        H[j+1,j] = la.norm(Q[:,j+1])
        Q[:,j+1] /= H[j+1,j]
        y,res,rank,s = la.lstsq(H[:j+2,:j+1], e[:j+2])
        toplot.append(res)
        j += 1
        
    evals = la.eig(A)[0]
    
    domain = np.linspace(0, len(toplot), len(toplot))
    plt.yscale('log')
    plt.subplot(1,2,1)
    plt.scatter(evals.real, evals.imag)
    plt.subplot(1,2,2)
    plt.yscale('log')
    plt.plot(domain, toplot)
    plt.show()
    
    
def problem2():
    '''Create the function for problem 2 which calls plot_gmres on An for n = -4,-2,0,2,4.
        Print out an explanation of how the convergence of the GMRES algorithm
        relates to the eigenvalues.
        
    '''

    values = [-4,-2,0,2,4]
    m = 200
    for n in values:
        print 'plotting n =' , n
        A = n * np.eye(m) + np.random.normal(0,1./(2 * np.sqrt(m)),(m,m))
        plot_gmres(A,np.ones(m),np.zeros(m))
    
    print 'It converges slowest when rhere is more divergence in the Eigenvalues.'
        
    
    
def gmres_k(A, b, x0, k=5, tol=1E-8, restarts=50):
    '''Use the GMRES(k) algorithm to approximate the solution to Ax=b.
    INPUTS:
    A - A callable function that calculates Ax for any vector x.
    b - A NumPy array.
    x0 - An arbitrary initial guess.
    k - Maximum number of iterations of the GMRES algorithm before
    restarting. Defaults to 5.
    tol - Stop iterating if the residual is less than 'tol'. Defaults
    to 1E-8.
    restarts - Maximum number of restarts. Defaults to 50.
    RETURN:
    Return (y, res) where 'y' is an approximate solution to Ax=b and 'res'
    is the residual.
    '''
    
    n = 0
    while n <= restarts:
        y,res=gmres(A,b,x0,k)
        if res<tol:
            return y,res
            x0=y
            n+=1
        return y, res
