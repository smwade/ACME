# Name this file 'solutions.py'.
"""Volume II: Compressed Sensing.
Sean Wade
Volume 2
2/15/2016
"""

import numpy as np
from matplotlib import pyplot as plt
from cvxopt import solvers, matrix
from camera import Camera
from visualize2 import visualizeEarth
from numpy import linalg as la

# Problem 1
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Return only the solution x (not any slack variable), as a flat NumPy array.

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        x ((n, ) ndarray): The solution to the minimization problem.
    """
    m, n = A.shape
    b = matrix(b.astype(float))

    c = matrix(np.hstack((np.ones(n), np.zeros(n))))
    G = np.hstack((-np.eye(n), np.eye(n)))
    Gi = np.hstack((-np.eye(n), -np.eye(n)))
    G = matrix(np.vstack((G,Gi)))

    h = matrix(np.zeros(2*n))
    A = matrix(np.hstack((np.zeros((m,n)), A)))

    sol = solvers.lp(c, G, h, A, b)
    return sol['x'][n:]


# Problem 2
def prob2(filename='ACME.png'):
    """Reconstruct the image in the indicated file using 100, 200, 250,
    and 275 measurements. Seed NumPy's random number generator with
    np.random.seed(1337) before each measurement to obtain consistent
    results.

    Resize and plot each reconstruction in a single figure with several
    subplots (use plt.imshow() instead of plt.plot()). Return a list
    containing the Euclidean distance between each reconstruction and the
    original image.
    """
    acme = 1 - plt.imread('ACME.png')[:,:,0]
    measures = [100, 200, 250, 275]
    results = []
    for i,m in enumerate(measures):
        plt.subplot(2,2,i+1)
        np.random.seed(1337)
        A = np.random.randint(low=0, high=2, size=(m, 32**2))
        b = A.dot(acme.flatten())
        rec = l1Min(A,b)
        rec = np.array(rec).reshape(acme.shape)
        plt.imshow(rec)
    plt.show()
    return results


# Problem 3
def prob3(filename="StudentEarthData.npz"):
    """Reconstruct single-pixel camera color data in StudentEarthData.npz
    using 450, 650, and 850 measurements. Seed NumPy's random number generator
    with np.random.seed(1337) before each measurement to obtain consistent
    results.

    Return a list containing the Euclidean distance between each
    reconstruction and the color array.
    """
    measures = [250, 400, 550]
    dist = []
    error = []

    earth = np.load(filename)
    faces, vert, C, V = earth['faces'],earth['vertices'], earth['C'], earth['V']
    camera = Camera(faces, vert, C)

    for m in measures:
        camera.add_lots_pic(m)
        A, B = camera.returnData()
        for col in xrange(B.shape[1]):
            b_hat = B[:,col]
            A_hat = np.dot(A, V)
            c = l1Min(A_hat, b_hat)
            c_hat = np.dot(V, c)
            if col == 0:
                c_stack = c_hat
            else:
                c_stack = np.column_stack((c_stack, c_hat))

        visualizeEarth(faces, vert, c_stack.clip(0,1))
        error.append(la.norm(c_stack - C))
    visualizeEarth(faces, vert, C)
    return error

prob3()

