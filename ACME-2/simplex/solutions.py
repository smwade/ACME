# name self file 'solutions.py'.
"""Volume II Lab 16: Simplex
Sean Wade
Math 324
<date>

Problems 1-6 give instructions on how to build the SimplexSolver class.
The grader will test your class by solving various linear optimization
problems and will only call the constructor and the solve() methods directly.
Write good docstrings for each of your class methods and comment your code.

prob7() will also be tested directly.
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """


    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        self.A = A
        self.b = b
        self.c = c
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.topVar = A.shape[1]
        self.sideVar = A.shape[0]
        self.basic = range(0,self.n)
        self.nonBasic = range(self.n, self.m + self.n)
        print self.basic
        print self.nonBasic


        m = A.shape[0]
        n = A.shape[1]

        origin = np.zeros(n)
        result = np.dot(A, origin)
        if np.any(np.greater(result, b)):
            raise ValueError()

        #part 2
        self.L = range(n, m+n) + range(0,n)
        self.initTableau()
        print self.T
        self.solve()

   
    # Sets uo self.T
    def initTableau(self):
        I = np.identity(self.m)
        A_bar = np.hstack((self.A,I))
        c_bar = -1 * np.hstack((self.c, np.zeros(self.m)))
        self.T = np.column_stack((self.b, A_bar, np.zeros(self.m)))
        top = np.hstack(([0], c_bar, [1]))
        self.T = np.vstack((top, self.T))
        self.m = self.T.shape[0]
        self.n = self.T.shape[1]

    def findPivot(self):
        enterIndex = -1
        for i in xrange(1, self.T.shape[1]):
            if (self.T[0][i] < 0):
                enterIndex = i
                break

        if enterIndex == -1:
            return "DONE"

        indexCol = np.copy(self.T[1:,enterIndex])
        
        # check if all negative
        if np.all(indexCol <= 0):
            raise ValueError("everything is negative")
        ratioVector = np.copy(self.T[1:,0]) / indexCol
        ratioVector[ratioVector <= 0] = float('inf')
        pivotRow = np.argmin(ratioVector) + 1
        fromBasic = enterIndex - 1
        toBasic = pivotRow - 1
        print self.basic[fromBasic]
        self.basic[self.basic.index(fromBasic)], self.nonBasic[toBasic] = \
                self.nonBasic[toBasic], self.basic[self.basic.index(fromBasic)]
        print self.L
        print enterIndex 
        return (enterIndex, pivotRow)


    def pivotToTableCol(self, pCol):
        return self.L.index(pCol) - self.topVar + 2

    def pivot(self, pivotTuple):
        pRowIndex = pivotTuple[1]
        pColIndex = pivotTuple[0]
        self.T[pRowIndex,:] = self.T[pRowIndex,:] / self.T[pRowIndex, pColIndex]
        for i in xrange(self.m):
            if i != pRowIndex:
                self.T[i,:] = self.T[i,:] - (self.T[i,pColIndex] * self.T[pRowIndex,:])
        print self.T



    def solve(self):
        """Solve the linear optimization problem.
        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while True:
            pivotIndex = self.findPivot()
            if pivotIndex == "DONE":
                break;
            else:
                self.pivot(pivotIndex)
        myDict = dict(zip(self.L, self.T[0,1:]))
        maxVal = self.T[0][0]
        return (maxVal, dict(zip(self.nonBasic, self.T[1:,0])), dict(zip(self.basic, self.T[0,1:self.n])))


def test():
    A = np.array([[1,-1],[3,1], [4,3]])
    b = np.array([2,5,7])
    c = np.array([3,2])
    return SimplexSolver(c, A, b).solve()
    

# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    data = np.load('productMix.npz')
    A_f = data['A']
    p = data['p']
    m = data['m']
    d = data['d']
    b = np.hstack((m,d))
    A = np.vstack((A_f, np.identity(4)))
    c = p
    r, t, y = SimplexSolver(c, A, b).solve()
    return t[0], t[1], t[2], t[3]





# END OF FILE =================================================================
