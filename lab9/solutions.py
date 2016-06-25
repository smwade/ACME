# #Image Compression (SVD)
# ##Sean Wade


from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.pylab as pylab



def truncated_svd(A,r=None,tol=10**-6):
    """Computes the truncated SVD of A. If r is None or equals the number 
        of nonzero singular values, it is the compact SVD.
    Parameters:
        A: the matrix
        r: the number of singular values to use
        tol: the tolerance for zero
    Returns:
        U - the matrix U in the SVD
        s - the diagonals of Sigma in the SVD
        Vh - the matrix V^H in the SVD
    """
    # Full SVD:  mxm mxn nXn
    # Truncated SVD:   mxr rxr rxn
    m, n = A.shape
    if r > n:
        r = n
    e_val, e_vec = la.eig(np.dot(A.conjugate().T, A))
    sing_val = np.sqrt(e_val)[:r]
    sort_index = np.argsort(sing_val)[::-1]
    S = e_val[sort_index]
    V = e_vec[:,sort_index]

    U = np.zeros((m, r))
    for i in xrange(r):
        U[:,i] = ((1/S[i]) * (A.dot(np.vstack(V[:,i])))).T
    
    S = np.diag(S)
    Vh = V.conjugate().T
    return U, S, Vh 

    

def visualize_svd():
    """Plot each transformation associated with the SVD of A."""
    A = np.matrix([[3,1],[1,3]])
    U, SIGMA, V_T = la.svd(A)
    SIGMA = np.diag(SIGMA)
    circle_points = np.load('circle.npz')['circle']
    unit_vectors = np.load('circle.npz')['unit_vectors']
    
    # PLOT THE CHANGE ON THE UNIT CIRCLE
    #-----------------------------------------------
    plt.subplot(221)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.plot(circle_points[0,:], circle_points[1,:])
    plt.plot(unit_vectors[0,:], unit_vectors[1,:])
    
    plt.subplot(222)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    V_T_S = np.dot(V_T, circle_points)
    unit_V_T_S = np.dot(V_T, unit_vectors)
    plt.plot(V_T_S[0,:], V_T_S[1,:])
    plt.plot(unit_V_T_S[0,:], unit_V_T_S[1,:])
    
    plt.subplot(223)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    S_V_T_S = np.dot(SIGMA, V_T_S)
    unit_S_V_T_S = np.dot(SIGMA, unit_V_T_S)
    plt.plot(S_V_T_S[0,:], S_V_T_S[1,:])
    plt.plot(unit_S_V_T_S[0,:], unit_S_V_T_S[1,:])
    
    plt.subplot(224)
    U_S_V_T_S = np.dot(U, S_V_T_S)
    unit_U_S_V_T_S = np.dot(U, unit_S_V_T_S)
    plt.plot(U_S_V_T_S[0,:], U_S_V_T_S[1,:])
    plt.plot(unit_U_S_V_T_S[0,:], unit_U_S_V_T_S[1,:])
    
    plt.suptitle("SVD Visualization")
    plt.show()
    


def svd_approx(A, k):
    """Returns best rank k approximation to A with respect to the induced 2-norm.
    
    Inputs:
    A - np.ndarray of size mxn
    k - rank 
    
    Return:
    Ahat - the best rank k approximation
    """
    U, S, Vh = la.svd(A, full_matrices=False)
    S = np.diag(S[:k])
    Ahat = U[:,:k].dot(S).dot(Vh[:k,:])
    return Ahat



def lowest_rank_approx(A,e):
    """Returns the lowest rank approximation of A with error less than e 
    with respect to the induced 2-norm.
    
    Inputs:
    A - np.ndarray of size mxn
    e - error
    
    Return:
    Ahat - the lowest rank approximation of A with error less than e.
    """
    U, S, Vh = la.svd(A, full_matrices=False)
    i = 0
    for sig_val in S:
        if sig_val < e:
            k = i
        else:
            i += 1
            
    return svd_approx(A, k)


def compress_image(filename,k):
    """Plot the original image found at 'filename' and the rank k approximation
    of the image found at 'filename.'
    
    filename - jpg image file path
    k - rank
    """
    # Black and White
    #X = plt.imread(filename)[:,:,0].astype(float)
    #aprox = svd_approx(X,k)
    #aprox[aprox > 255] = 255
    #aprox[aprox < 0] = 0
    #plt.subplot(121)
    #plt.axis("off")
    #plt.title("Original Image")
    #plt.imshow(X, cmap=cm.gray)
    #plt.subplot(122)
    #plt.axis("off")
    #plt.title("Rank %s Approximation" % k)
    #plt.imshow(aprox, cmap=cm.gray)
    
    
    
    
    IMG = plt.imread(filename)
    R_IMG_APROX = svd_approx(IMG[:,:,0].astype(float), k)
    G_IMG_APROX = svd_approx(IMG[:,:,1].astype(float), k)
    B_IMG_APROX = svd_approx(IMG[:,:,2].astype(float), k)
    R_IMG_APROX[R_IMG_APROX > 255] = 255
    R_IMG_APROX[R_IMG_APROX < 0] = 0
    G_IMG_APROX[G_IMG_APROX > 255] = 255
    G_IMG_APROX[G_IMG_APROX < 0] = 0
    B_IMG_APROX[B_IMG_APROX > 255] = 255
    B_IMG_APROX[B_IMG_APROX < 0] = 0
    imag_aprox = np.dstack([R_IMG_APROX.astype(np.uint8), G_IMG_APROX.astype(np.uint8), B_IMG_APROX.astype(np.uint8)])
    plt.subplot(121)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(IMG)
    plt.subplot(122)
    plt.axis("off")
    plt.title("Rank %s Approximation" % k)
    plt.imshow(imag_aprox)
    
    
