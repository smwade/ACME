# Sean Wade
'''
Lab 14 - Newton's Method.
'''

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

def Newtons_method(f, x0, Df, iters=15, tol=.002):
    '''Use Newton's method to approximate a zero of a function.
    
    INPUTS:
    f     - A function handle. Should represent a function from 
            R to R.
    x0    - Initial guess. Should be a float.
    Df    - A function handle. Should represent the derivative 
            of `f`.
    iters - Maximum number of iterations before the function 
            returns. Defaults to 15.
    tol   - The function returns when the difference between 
            successive approximations is less than `tol`.
    
    RETURN:
    A tuple (x, converged, numiters) with
    x           - the approximation to a zero of `f`
    converged   - a Boolean telling whether Newton's method 
                converged
    numiters    - the number of iterations the method computed
    '''
    i = 0
    converged = False
    diff = float('inf')
    xold = x0
    xnew = x0
    while i < iters and diff > tol:
        xold = xnew
        xnew = xold - f(xold)/Df(xold)
        diff = abs(xold - xnew)
        i += 1
        if diff < tol:
            converged = True

    return (xnew, converged, i)


def prob2():
    '''
    Print the answers to the questions in problem 2.
    '''
    # 1
    f = lambda x: np.cos(x)
    Df = lambda x: -np.sin(x)
    x0 = 1
    iters = 100

    # 2
    f = lambda x: np.sin(x) / x - x
    x = np.linspace(-4,4)
    plt.plot(x,f(x))
    plt.show()
    Df = lambda x: ((x*np.cos(x) - np.sin(x)) / x**2) - 1

    #3
    f = lambda x: x**9
    x0 = 1
    Df = lambda x: 9*x**8
    tol = 1e-5
    Newtons_method(f,x0,Df,500,tol)

    #4
    f = lambda x: np.sign(x)*np.power(np.abs(x), 1./3)
    x0 = .01
    Df = lambda x: (1./3)*x**(-2./3)

    print '1. It takes 4 and 3 iterations'
    print '2. 0.8767262'
    print '3. 81, it is so slow because the derivative is really small'
    print '4. It diverges because it is not in C1'

def Newtons_2(f, x0, iters=15, tol=.002):
    '''
    Optional problem.
    Re-implement Newtons method, but without a derivative.
    Instead, use the centered difference method to estimate the derivative.
    '''
    raise NotImplementedError('Newtons Method 2 not implemented')

def plot_basins(f, Df, roots, xmin, xmax, ymin, ymax, numpoints=100, iters=15, colormap='brg'):
    '''Plot the basins of attraction of f.
    
    INPUTS:
    f       - A function handle. Should represent a function 
            from C to C.
    Df      - A function handle. Should be the derivative of f.
    roots   - An array of the zeros of f.
    xmin, xmax, ymin, ymax - Scalars that define the domain 
            for the plot.
    numpoints - A scalar that determines the resolution of 
            the plot. Defaults to 100.
    iters   - Number of times to iterate Newton's method. 
            Defaults to 15.
    colormap - A colormap to use in the plot. Defaults to 'brg'. 
    
    RETURN:
    Returns nothing, but should display a plot of the basins of attraction.
    '''
    def match(x):
        return np.argmin(abs(roots - x))

    xreal = np.linspace(xmin, xmax, numpoints) 
    ximag = np.linspace(ymin, ymax, numpoints) 
    Xreal, Ximag = np.meshgrid(xreal, ximag)
    Xold = Xreal+1j*Ximag
    Xnew = Xold
    for _ in xrange(iters):
        Xold = Xnew
        Xnew = Xold - f(Xold)/Df(Xold)

    match_vector = np.vectorize(match)
    Xnew_new = match_vector(Xnew)

    plt.pcolormesh(Xreal, Ximag, Xnew_new,cmap=colormap)
    plt.show()

# f = lambda x: x**3-x
# Df = lambda x: 3*x**2 - 1
# plot_basins(f, Df, np.array([-1,0,1]), -1.5, 1.5, -1.5, 1.5, 1000, 15, 'brg')

def prob5():
    '''
    Using the function you wrote in the previous problem, plot the basins of
    attraction of the function x^3 - 1 on the interval [-1.5,1.5]X[-1.5,1.5]
    (in the complex plane).
    '''
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    roots = np.array([1, -1j**(1./3), 1j**(2./3)])
    plot_basins(f, Df, roots, -1.5, 1.5, -1.5, 1.5, 1000, 15, 'brg')
