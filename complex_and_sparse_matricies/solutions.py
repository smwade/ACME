# Sean Wade
# Lab 4: Volume 1
# 09/22/15

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sl
import timeit

'''Functions for use in problem 1.'''
# Run through a single for loop.
def func1(n):
    n = 500*n
    sum(xrange(n))

# Run through a double for loop.
def func2(n):
    n = 3*n
    t = 0
    for i in xrange(n):
        for j in xrange(i):
            t += j

# Square a matrix.
def func3(n):
    n = int(1.2*n)
    A = np.random.rand(n, n)
    np.power(A, 2)

# Invert a matrix.
from scipy import linalg as la
def func4(n):
    A = np.random.rand(n, n)
    la.inv(A)

# Find the determinant of a matrix.
from scipy import linalg as la
def func5(n):
    n = int(1.25*n)
    A = np.random.rand(n, n)
    la.det(A)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped



def Problem1():
    """Create a plot comparing the times of func1, func2, func3, func4, 
    and func5. Time each function 4 times and take the average of each.
    """
    list1 = []
    test_values = [100, 200, 400, 800]
    for x in test_values:
        wrapped = wrapper(func1, x)
        list1.append(timeit.timeit(wrapped, number=10))

    list2 = []
    for x in test_values:
        wrapped = wrapper(func2, x)
        list2.append(timeit.timeit(wrapped, number=10))

    list3 = []
    for x in test_values:
        wrapped = wrapper(func3, x)
        list3.append(timeit.timeit(wrapped, number=10))

    list4 = []
    for x in test_values:
        wrapped = wrapper(func4, x)
        list4.append(timeit.timeit(wrapped, number=10))

    list5 = []
    for x in test_values:
        wrapped = wrapper(func5, x)
        list5.append(timeit.timeit(wrapped, number=10))

    plt.plot(test_values, list1, label='Function 1')
    plt.plot(test_values, list2, label='Function 2')
    plt.plot(test_values, list3, label='Function 3')
    plt.plot(test_values, list4, label='Function 4')
    plt.plot(test_values, list5, label='Function 5')
    plt.legend(loc='upper left')
    plt.show()


def Problem2(n):
    mdiag = np.linspace(2,2,n)
    ldiag = np.linspace(-1,-1,n)

    diags = np.array([-1, 0, 1])
    data = np.array([ldiag, mdiag, ldiag])

    sol = sparse.spdiags(data, diags, n, n, format='csr')     
    return sol

def Problem3(n):
    """Generate an nx1 random array b and solve the linear system Ax=b
    where A is the tri-diagonal array in Problem 2 of size nxn
    """
    b = np.random.rand(n,1) 
    a = Problem2(n)
    result = sl.spsolve(a,b)
    return result

def Problem4(n, sparse=False):
    """Write a function that accepts an integer argument n and returns
    (lamba)*n^2 where (lamba) is the smallest eigenvalue of the sparse 
    tri-diagonal array you built in Problem 2.
    """
    A = Problem2(n)
    
    l, x = sl.eigs(A.asfptype(),k=n-2, which='SM')
    eigen_min = np.min(l)
    res = eigen_min*n**2
    return res

    

