# matrix_multiply.py
"""Volume II Lab 1: The Standard Library (Auxiliary File)
Use this module to complete problem 4. Do not modify.
"""

# numpy, short for "numerical python", will be introduced in Volume I Lab 2
import numpy as np

def load_matrices(filename):
    """Returns two matrices if the correct filename is given."""
    
    files = np.load(filename)
    return files['arr_0'], files['arr_1']


def method1(A,B):
    """Multiply the matrices 'A' and 'B' together using nested for loops."""
    
    product_matrix = np.zeros((A.shape[0], B.shape[1]))
    for i in range(product_matrix.shape[0]):
        for j in range(product_matrix.shape[1]):
            for k in range(product_matrix.shape[0]):
                product_matrix[i,j] += A[i,k]*B[k,j] 
    
    return product_matrix


def method2(A,B):
    """Multiply the matrices 'A' and 'B' together with some vectorization.
    Use xrange() instead of range() to make things a little faster.
    """
    
    product_matrix = np.zeros((A.shape[0], B.shape[1]))
    for i in xrange(product_matrix.shape[0]):
        for j in xrange(product_matrix.shape[1]):
            product_matrix[i,j] = np.dot(A[i,:], B[:,j])
    
    return product_matrix

def method3(A,B):
    """Use numpy's matrix multiplication method for maximum speed."""
    
    return np.dot(A,B)
    
# ============================== END OF FILE ================================ #