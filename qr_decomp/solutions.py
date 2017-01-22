# solutions.py
"""Volume I Lab 6: QR Decomposition.
Sean Wade
13/10/2015
"""

import numpy as np
from math import sqrt
from scipy import linalg as la


A = np.matrix([[1,2,3], [1,1,1], [5,6,8], [2,3,5]])

def QR(A):
    '''
    Compute the QR decomposition of a matrix.
    Accept an m by n matrix A of rank n. 
    Return Q, R
    '''
    A.astype(float)
    m, n = A.shape
    Q = np.copy(A).astype(float)
    R = np.zeros((n,n)).astype(float)
    for i in xrange(0, n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        for j in xrange(i+1, n):
            R[i,j] = np.dot(np.transpose(Q[:,j]), Q[:,i])
            # R[i,j] = np.inner(Q[:,j], Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q, R
    
def prob2(A):
    '''
    Use your QR decomposition from the previous problem to compute 
    the determinant of A.
    Accept a square matrix A of full rank.
    Return |det(A)|.
    '''
    # since det(QR) = det(Q)det(R) = det(R) is multiply on the trace

    Q, R = QR(A)
    X = R.diagonal()
    print R
    print X
    return abs(np.prod(X))



def householder(A):
    '''
    Use the Householder algorithm to compute the QR decomposition
    of a matrix.
    Accept an m by n matrix A of rank n. 
    Return Q, R
    '''
    A = A.astype(float)
    m,n = A.shape
    R = np.copy(A)
    Q = np.identity(m)
    for k in xrange(0, n):
        u = np.copy(R[k:,k])
        np.reshape(u, ((m-k), 1))
        u[0] = u[0] + -(u[0]/-u[0]) * la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u,u).dot(R[k:,k:])
        Q[k:] = Q[k:] - 2*np.outer(u,u).dot(Q[k:])

    return np.transpose(Q), R

        

def hessenberg(A):
    '''
    Compute the Hessenberg form of a matrix. Find orthogonal Q and upper
    Hessenberg H such that A = QtHQ.
    Accept a non-singular square matrix A.
    Return Q, H
    '''

    A = A.astype(float)
    m,n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)
    for k in xrange(0, n-2):
        u = np.copy(H[k+1:,k])
        u[0] = u[0] + (-(u[0]/(-u[0])))*la.norm(u)
        u = u/la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,u).dot(H[k+1:,k:])
        H[:,k+1:] = H[:,k+1:] - 2*np.dot(H[:,k+1:], np.outer(u,u))
        Q[k+1:] = Q[k+1:] - 2*np.outer(u,u).dot(Q[k+1:])
    return Q,H


Q,H = householder(A)
print Q
print H
print np.dot(Q, H)



def givens(A):
    '''
    EXTRA 20% CREDIT
    Compute the Givens triangularization of matrix A.
    Assume that at the ijth stage of the algorithm, a_ij will be nonzero.
    Accept A
    Return Q, R
    '''
#    A = A.astype(float)
#    m,n = np.shape(A)
#    R = np.copy(A)
#    Q = np.identity(m)
#    G = np.empty((2,2))
#    for j in xrange(0, n):
#        for i in xrange(m-1, j+2):
#            a, b = R[i-1,j], R[i,j]
#            G = [[a,b],[-b,a]]/sqrt(a**2 + b**2)
#            R[i-1:i+1,j:] = np.dot(G, R[i-1:i+1, j:])
#            Q[i-1:i+1, :] = np.dot(G, Q[i-1:i+1,:])
#    return np.transpose(Q), R
#
#Q, R = givens(A)
#print Q
#print R
#print np.dot(Q, R)


def prob6(H):
    '''
    EXTRA 20% CREDIT
    Compute the Givens triangularization of an upper Hessenberg matrix.
    Accept upper Hessenberg H.
    
    '''
    pass
