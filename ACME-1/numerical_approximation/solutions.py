# Sean Wade
# Lab 12
# December 1, 2015

import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

# Problem 1: Implement this function.
def centered_difference_quotient(f,pts,h = 1e-5):
    '''
    Compute the centered difference quotient for function (f)
    given points (pts).
    Inputs:
        f (function): the function for which the derivative will be approximated
        pts (array): array of values to calculate the derivative
    Returns:
        centered difference quotient (array): array of the centered difference
            quotient
    '''
    return (.5 * f(pts + h) - .5 * f(pts - h)) / h
    


# Problem 2: Implement this function.
def jacobian(f,n,m,pt,h = 1e-5):
    '''
    Compute the approximate Jacobian matrix of f at pt using the centered
    difference quotient.
    Inputs:
        f (function): the multidimensional function for which the derivative
            will be approximated
        n (int): dimension of the domain of f
        m (int): dimension of the range of f
        pt (array): an n-dimensional array representing a point in R^n
        h (float): a float to use in the centered difference approximation
    Returns:
        Jacobian matrix of f at pt using the centered difference quotient.
    '''
    jacobian = np.zeros((m,n))
    I = np.eye(n)
    for j in xrange(n):
        jacobian[:,j] = (.5 * f(pt + h*I[:,j]) - .5 * f(pt - h*I[:,j])) / h

    return jacobian


def test():
    f = lambda x: np.array([x[0]**2 * x[1], 5*x[0] + np.sin(x[1])])
    n = 2
    m = 2
    pt = np.array([1,0])
    print jacobian(f,n,m,pt)


# Problem 3: Implement this function.
def findError():
    '''
    Compute the maximum error of your jacobian function for the function
    f(x,y)=[(e^x)*sin(y)+y^3,3y-cos(x)] on the square [-1,1]x[-1,1].
    Returns:
        Maximum error of your jacobian function.
    '''
    f = lambda x: np.array([np.exp(x[0])*np.sin(x[1]) + x[1]**3, 3*x[1]-np.cos(x[0])])
    f_derivative = lambda x: np.array([[np.exp(x[0]) * np.sin(x[1]), np.exp(x[0])*np.cos(x[1])+3*x[1]**2], [np.sin(x[1]), 3]])
    L = np.linspace(-1,1,100)
    worst_error = 0
    for y in L:
        for x in L:
            approx = jacobian(f,2,2,np.array([x,y]))
            real = f_derivative(np.array([x,y]))
            error = la.norm(real - approx)
            if error > worst_error:
                worst_error = error
    return worst_error



G = 1./159 * np.array(
            [[2,4,5,4,2], 
            [4,9, 12,9,4], 
            [5,12,15,12,5], 
            [4,9, 12,9,4], 
            [2,4, 5 ,4,2]])       
# Problem 4: Implement this function.
def Filter(image,F=G):
    '''
    Applies the filter to the image.
    Inputs:
        image (array): an array of the image
        F (array): an nxn filter to be applied (a numpy array).
    Returns:
        The filtered image.
    '''
    h,k = F.shape
    m,n = image.shape
    dif = h/2
    image_pad = np.zeros((m+h, n+h))
    image_pad[dif:dif+m, dif:dif+n] = image
    C = np.zeros(image.shape)
    for i in xrange(m):
        for j in xrange(n):
            C[i,j] = np.trace(np.dot(image_pad[i:i+h,j:j+h], F.T))
    return C

def problem_4():
    image = plt.imread('cameraman.png')
    for _ in xrange(4):
        image = Filter(image)
    plt.imshow(image, cmap='gray')
    plt.show()


# Problem 5: Implement this function.
def sobelFilter(image):
    '''
    Applies the Sobel filter to the image
    Inputs:
        image(array): an array of the image in grayscale
    Returns:
        The image with the Sobel filter applied.
    '''
    S = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).astype(float)

    d_x = Filter(image, S)
    d_y = Filter(image, S.T)

    DA = (d_x**2 + d_y**2)**(.5)
    M = DA.mean() * 4
    Outline = np.copy(DA)
    Outline[DA>M] = 1
    Outline[DA<=M] = 0
    return Outline 

def problem_5():
    A = sobelFilter(plt.imread('cameraman.png'))
    plt.imshow(A, cmap='gray')
    plt.show()
