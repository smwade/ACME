import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
from sympy import subfactorial
from math import factorial
from scipy import linalg as la

##############  Problem 1  ##############
def prob1():
    '''
    Randomly perturb w_coeff by replacing each coefficient a_i with a_i*r_i, where
    r_i is drawn from a normal distribution centered at 1 with varience 1e-10.
    	
    Plot the roots of 100 such experiments in a single graphic, along with the roots
    of the unperturbed polynomial w(x)
    	
    Using the final experiment only, estimate the relative and absolute condition number
    (in any norm you prefer).
    	
    RETURN:
    Should display graph of all 100 perturbations.
    Should print values of relative and absolute condition.
    '''
    w_roots = np.arange(1, 22)
    w_coeffs = np.array([1, -210, 20615, -1256850, 53327946, -1672280820,
				40171771630, -756111184500, 11310276995381,
    				-135585182899530, 1307535010540395,
    				-10142299865511450, 63030812099294896,
    				-311333643161390640, 1206647803780373360,
    				-3599979517947607200, 8037811822645051776,
    				-12870931245150988800, 13803759753640704000,
    				-8752948036761600000, 2432902008176640000])
        
    for _ in xrange(0,100):    
        r = normal(1.0, 1e-10)
        i = np.random.randint(0,21)
        peturb = np.ones(21)
        peturb[i] = r
        peturb = peturb * w_coeffs
        pr = np.roots(np.poly1d(peturb))
        x = np.real(pr)
        y = np.imag(pr)
        plt.scatter(x,y,s=.9)
    plt.scatter(np.roots(np.poly1d(w_coeffs)), np.zeros(20), c='r')
    plt.show()
    perturb = np.zeros(21)
    w_roots = np.sort(w_roots)
    perturbed_roots = np.sort(peturb)
    peturb[peturb == 1] = 0
    con = la.norm(perturbed_roots-w_roots)/la.norm(peturb)
    # relative condition
    k = la.norm(perturbed_roots-w_roots, np.inf)/la.norm(perturbed_roots, np.inf)
    rel = k*la.norm(w_coeffs, np.inf)/la.norm(w_roots, np.inf)
    print "the absolute is: %s" % con
    print "the ralative is: %s" % rel



##############  Problem 2  ##############	
def eig_condit(M):
    '''
    Approximate the condition number of the eigenvalue problem at M.
    
    INPUT:
    M - A 2-D square NumPy array, representing a square matrix.
    
    RETURN:
    A tuple containing approximations to the absolute and 
    relative condition numbers of the eigenvalue problem at M.
    '''
    eigs = la.eig(M)[0]
    peturb = np.random.normal(0, 1e-10, M.shape) + np.random.normal(0, 1e-10, M.shape) * 1j
    eigsp = la.eig(M + peturb)[0]
    # absolute condition number
    k = la.norm(eigs - eigsp)/la.norm(peturb)
    # relative condition number
    rel = k*la.norm(M)/la.norm(eigs)
    return (k, rel)

# m = np.array([[1.,500000000.], [.5, 2.]])
# print eig_condit(m)
print "problem 2"
print "A large condition number is the matrix [[1.,500000000.], [.5, 2.]].  it returned the values (4383.1509177915877, 98010233.588174656)"
print "When this is made symetric the relative number is much smaller and on the order of 1"
#   1 pt extra credit
def plot_eig_condit(x0=-100, x1=100, y0=-100, y1=100, res=10):
    '''
    Create a grid of points. For each pair (x,y) in the grid, find the 
    relative condition number of the eigenvalue problem, using the matrix 
    [[1 x]
     [y 1]]
    as your input. You can use plt.pcolormesh to plot the condition number
    over the entire grid.
    
    INPUT:
    x0 - min x-value of the grid
    x1 - max x-value
    y0 - min y-value
    y1 - max y-value
    res - number of points along each edge of the grid
    '''
    raise NotImplementedError('plot_eig_condit not implemented')

##############  Problem 3  ##############
def integral(n):
    '''
    RETURN I(n)
    '''
    return (-1)**n * subfactorial(n) + (-1)**(n+1) * factorial(n) / np.e

def prob3():
    '''
    For the values of n in the problem, compute integral(n). Compare
    the values to the actual values, and print your explanation of what
    is happening.
    '''
    
    #actual values of the integral at specified n
    actual_values = [0.367879441171, 0.145532940573, 0.0838770701034, 
                 0.0590175408793, 0.0455448840758, 0.0370862144237, 
                 0.0312796739322, 0.0270462894091, 0.023822728669, 
                 0.0212860390856, 0.0192377544343] 
    x_vals = [1,5,10,15,20,25,30,35,40,45,50]
    approx_values = map(integral, x_vals)
    for x in xrange(len(actual_values)):
         print abs(actual_values[x] - approx_values[x])
    print "THe values are pretty far apart with low number of n, but as the number increases it gets a smaller error because the larger the number the more acurate"

