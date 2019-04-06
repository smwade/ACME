"""Volume II Lab 11: Wavelets
Sean Wade
Math 320
11/19/2015
"""

import numpy as np
import scipy as sp
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

# Problem 1: Implement this AND the following function.
def dwt(X, L, H, n):
    """Compute the Discrete Wavelet Transform of X using filters L and H.

    Parameters:
        X (1D ndarray): The signal to be processed.
        L (1D ndarray): The low-pass filter.
        H (1D ndarray): The high-pass filter.
        n (int > 0): Controls the degree of transformation.
    
    Returns:
        a list of the wavelet decomposition arrays.
    """
    i = 0
    A_i = X
    D =  []
    while i < n:
        d_iplus1 = fftconvolve(A_i, H)[1::2]
        D.append(d_iplus1)
        A_i = fftconvolve(A_i, L)[1::2]
        i += 1

    D.append(A_i)
    return D[::-1]
        

def plot(X, L, H, n):
	"""Plot the results of dwt with the given inputs.
        Your plot should be very similar to Figure 2.

    Parameters:
        X (1D ndarray): The signal to be processed.
        L (1D ndarray): The low-pass filter.
        H (1D ndarray): The high-pass filter.
        n (int > 0): Controls the degree of transformation.

    """
        coeffs = dwt(X, L, H, n)
        points = len(coeffs)+1
        plt.subplot(points,1, 1)
        plt.plot(X)
        for i in xrange(1,points):
            plt.subplot(points, 1, i+1)
            plt.plot(coeffs[i-1])
        plt.show()


# Problem 2: Implement this function.
def idwt(coeffs, L, H):
    """
    Parameters:
        coeffs (list): a list of wavelet decomposition arrays.
        L (1D ndarray): The low-pass filter.
        H (1D ndarray): The high-pass filter.
    Returns:
        The reconstructed signal (as a 1D ndarray).
    """
    A = coeffs[0]
    D = coeffs[1]
    for i in xrange(len(coeffs)-1):
        upsample_A = np.zeros(2*A.size)
        upsample_A[::2] = A
        upsample_D = np.zeros(2*coeffs[i+1].size)
        upsample_D[::2] = coeffs[i+1]
        A = fftconvolve(upsample_A, L)[:-1] + fftconvolve(upsample_D, H)[:-1]
    return A

def problem_2_test():
    L = np.ones(2)/np.sqrt(2)
    H = np.array([1,-1])/np.sqrt(2)

    domain = np.linspace(0, 4*np.pi, 1024)
    noise = np.random.randn(1024)*.1
    niosysin = np.sin(domain) + noise
    coef = dwt(niosysin, L, H*-1, 4)
    result = idwt(coef, L, H)
    print np.allclose(niosysin, result)

def problem_1_test():
    L = np.ones(2)/np.sqrt(2)
    H = np.array([-1,1])/np.sqrt(2)

    domain = np.linspace(0, 4*np.pi, 1024)
    noise = np.random.randn(1024)*.1
    niosysin = np.sin(domain) + noise
    plot(niosysin, L, H, 4)
