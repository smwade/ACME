# Sean Wade
# Arnoldi Iteration

import numpy as np
from scipy import linalg as la
from scipy.fftpack import fft
from matplotlib import pyplot as plt

def arnoldi(b, Amul, k, tol=1E-8):
    '''Perform `k' steps of the Arnoldi iteration on the linear 
        operator defined by `Amul', starting with the vector 'b'.
    '''
    Q = np.empty((len(b), k+1), dtype=np.complex128)
    H = np.zeros((k+1, k), dtype=np.complex128)
    Q[:,0] = b / la.norm(b)
    for j in xrange(0, k):
        Q[:, j+1] = Amul(Q[:,j])
        for i in xrange(0, j+1):
            H[i,j] = np.vdot(Q[:,i].T, Q[:,j+1])
            Q[:,j+1] = Q[:,j+1]- H[i,j]*Q[:,i]
        H[j+1, j] = np.sqrt(np.dot(np.conjugate(Q[:,j+1]), Q[:,j+1]))
        if np.abs(H[j+1,j]) < tol:
            return H[:j+1,:j+1], Q[:,:j+1]
        Q[:,j+1] = Q[:,j+1] / H[j+1,j]
    return H[:-1,:], Q


def ritz(Amul, dim, k, iters):
    ''' Find `k' Ritz values of the linear operator defined by `Amul'.
    '''
    if iters > dim or iters < k:
        raise ValueError("bad imput")

    b = np.random.rand(dim)
    H, Q = arnoldi(b, Amul, iters)
    ritz = la.eig(H[:k,:k])[0]
    return ritz

def fft_eigs(dim=2**20,k=10):
    '''Return the largest k Ritz values of the Fast Fourier transform
        operating on a space of dimension dim.
    '''
    return ritz(fft, dim, k, k)


def plot_ritz(A, n, iters):
    ''' Plot the relative error of the Ritz values of `A'.
    '''
    Amul = A.dot
    b = np.random.rand(A.shape[0])
    Q = np.empty((len(b), iters+1), dtype = np.complex128)
    H = np.zeros((iters+1, iters), dtype = np.complex128)
    Q[:, 0] = b / la.norm(b)
    eigvals = np.sort(abs(la.eig(A)[0]))[::-1]
    eigvals = eigvals[:n]
    abs_err = np.zeros((iters,n))

    for j in xrange(iters):
        Q[:, j+1] = Amul(Q[:, j])
        for i in xrange(j+1):
            H[i,j] = np.vdot(Q[:,i].conjugate(), (Q[:, j+1]))
            Q[:,j+1] = Q[:,j+1] - H[i,j] * (Q[:,i])

        H[j+1, j] = np.sqrt(np.vdot(Q[:, j+1], Q[:, j+1].conjugate()))
        Q[:,j+1] = Q[:,j+1] / H[j+1, j]

        if j < n:
            rit = np.zeros(n, dtype = np.complex128)
            rit[:j+1] = np.sort(la.eig(H[:j+1, :j+1])[0])[::-1]
            abs_err[j,:] = abs(eigvals - rit) / abs(eigvals)
        else:
            rit = np.sort(la.eig(H[:j+1,:j+1])[0])[::-1]
            rit = rit[:n]
            abs_err[j,:] = abs(eigvals - rit) / abs(eigvals)

    for i in xrange(n):
        plt.semilogy(abs_err[:,i])
    plt.show()

#plot_ritz(np.random.rand(300, 300), 10, 175)



#if __name__ == '__main__':
    #A = np.array([[1,0,0],[0,2,0],[0,0,3]])
    # Amul = lambda x: A.dot(x)
    #H, Q = arnoldi(np.array([1,1,1]), Amul, 3)
    #print np.allclose(H, np.conjugate(Q.T).dot(A).dot(Q) )
    #print fft_eigs()
