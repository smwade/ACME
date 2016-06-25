# spec.py
"""Volume I Lab 5: Invertible Affine Transformations and Linear Systems.
Name: Sean Wade
Date: 09/29/15
"""

# include your import statements here.
import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from time import time
import math

# Helper Functions

def plot_transform(original, new):
    """Display a plot of points before and after a transform.
    
    Inputs:
        original (array) - Array of size (2,n) containing points in R2 as columns.
        new (array) - Array of size (2,n) containing points in R2 as columns.
    """
    window = [-5,5,-5,5]
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.gca().set_aspect('equal')
    plt.scatter(original[0], original[1])
    plt.axis(window)
    plt.subplot(1, 2, 2)
    plt.title('After')
    plt.gca().set_aspect('equal')
    plt.scatter(new[0], new[1])
    plt.axis(window)
    plt.show()

def type_I(A, i, j):  
    """Swap the i-th and j-th rows of A."""
    A[i], A[j] = np.copy(A[j]), np.copy(A[i])
    
def type_II(A, i, const):  
    """Multiply the i-th row of A by const."""
    A[i] *= const
    
def type_III(A, i, j, const):  
    """Add a constant of the j-th row of A to the i-th row."""
    A[i] += const*A[j]


# Problem 1
def dilation2D(A, x_factor, y_factor):
    """Scale the points in A by x_factor in the x direction and y_factor in
    the y direction. Returns the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        x_factor (float) - scaling factor in the x direction.
        y_factor (float) - scaling factor in the y direction.
    """
    transform = np.array([[x_factor, 0], [0, y_factor]])
    changed = np.dot(transform, A)
    return changed


# Problem 2
def rotate2D(A, theta):
    """Rotate the points in A about the origin by theta radians. Returns 
    the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        theta (float) - number of radians to rotate points in A.
    """
    transform = np.array([[math.cos(theta), -(math.sin(theta))], [math.sin(theta), math.cos(theta)]])
    changed = np.dot(transform, A)
    return changed
    
# Problem 3
def translate2D(A, b):
    """Translate the points in A by the vector b. Returns the new array.
    
    Inputs:
        A (array) - Array of size (2,n) containing points in R2 stored as columns.
        b (2-tuple (b1,b2)) - Translate points by b1 in the x direction and by b2 
            in the y direction.
    """
    transform = np.vstack([b[0], b[1]])
    changed = A + transform 
    return changed
   
# Problem 4
def p2_pos(speed, time, p2_vector):
    return ((speed*time)/ (math.sqrt(p2_vector[0]**2 + p2_vector[1]**2))) * p2_vector
def p1_pos(p1_start, time, omega, p2_pos, p2_vector, speed):
    return rotate2D(p1_start, time * omega) + p2_pos(speed, time, p2_vector)


def rotatingParticle(time, omega, direction, speed):
    """Display a plot of the path of a particle P1 that is rotating 
    around another particle P2.
    
    Inputs:
     - time (2-tuple (a,b)): Time span from a to b seconds.
     - omega (float): Angular velocity of P1 rotating around P2.
     - direction (2-tuple (x,y)): Vector indicating direction.
     - speed (float): Distance per second.
    """
    p2_vector = np.vstack([direction[0], direction[1]])
    p1_start = np.array([[1],[0]])

    x = []
    y = []

    time = np.linspace(time[0], time[1], 500)
    for t in time:
        p1 = p1_pos(p1_start, t, omega, p2_pos, p2_vector, speed)
        x.append(p1[0])
        y.append(p1[1])

    plt.plot(x,y)
    plt.show()

# rotatingParticle((0,10), math.pi, (1,1), 2)

    
# Problem 5
def REF(A):
    """Reduce a square matrix A to REF. During a row operation, do not
    modify any entries that you know will be zero before and after the
    operation. Returns the new array."""
    size = A.shape[0] 
    for x in range(0, size):
        for y in range(x+1, size):
            a = np.copy(A[y, x])
            b = np.copy(A[x,x])
            type_III(A, y, x, -a/b)

    return A

# Problem 6
def LU(A):
    """Returns the LU decomposition of a square matrix."""
    m,n = A.shape
    U = A.copy().astype(float)
    L = np.identity(n).astype(float)
    for i in range(1,n):
        for j in range(0, i):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] = U[i,j:] - L[i, j]*U[j,j:]
    return L, U

# Problem 7
def time_LU():
    """Print the times it takes to solve a system of equations using
    LU decomposition and (A^-1)B where A is 1000x1000 and B is 1000x500."""
    
    
    time_lu_factor = .0164574  # set this to the time it takes to perform la.lu_factor(A)
    time_inv = .05261012332 # set this to the time it takes to take the inverse of A
    time_lu_solve = .0146859  # set this to the time it takes to perform la.lu_solve()
    time_inv_solve = .00992234  # set this to the time it take to perform (A^-1)B


    print "LU solve: " + str(time_lu_factor + time_lu_solve)
    print "Inv solve: " + str(time_inv + time_inv_solve)
    
    # What can you conclude about the more efficient way to solve linear systems?
    print "LU is more efficiant"  # print your answer here.

