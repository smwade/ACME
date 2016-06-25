# Page Rank
# Sean Wade

import numpy as np
from scipy.sparse import dok_matrix
import scipy.linalg as la
from scipy.sparse import linalg as sla

def to_matrix(filename,n):
    '''
    Return the nxn adjacency matrix described by datafile.
    INPUTS:
    datafile (.txt file): Name of a .txt file describing a directed graph. 
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile
    RETURN:
        Return a SciPy sparse `dok_matrix'.
    '''
    values = []
    with open(filename, 'r') as myfile:
        for line in myfile:
            value = line.strip().split()
            try:
                new_val = [int(i) for i in value]
                values.append(new_val)
            except ValueError:
                pass
    adj = dok_matrix((n,n))
    for d in values:
        adj[d[0],d[1]] = 1

    return adj

def calculateK(A,n):
    '''
    Compute the matrix K as described in the lab.
    Input:
        A (array): adjacency matrix of an array
        N (int): the datasize of the array
    Return:
        K (array)
    '''
    D = np.empty(n)

    for i in range(n):
        if i not in A.nonzero()[0]:
            A[i]=1
        D[i]=len(A[i].nonzero()[0])
        A[i]/=D[i]

    return A.T
    

def iter_solve(adj, N=None, d=.85, tol=1E-5):
    '''
    Return the page ranks of the network described by `adj`.
    Iterate through the PageRank algorithm until the error is less than `tol'.
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N (int) - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    tol  - Stop iterating when the change in approximations to the solution is
        less than `tol'. Defaults to 1E-5.
    Returns:
    The approximation to the steady state.
    '''
    if N is None:   
        N = adj.shape[0]

    K = calculateK(adj,N)[:N,:N]
    ones = np.ones((N,1))
    p = 1./N*ones
    p1 = d*K.dot(p)+(1.-d)/N*ones
    while la.norm(p1-p)>=tol:
        p = p1
        p1 = d*K.dot(p)+(1.-d)/N*ones
    return p1.T/np.sum(p1.T)

    
def eig_solve( adj, N=None, d=.85):
    '''
    Return the page ranks of the network described by `adj`. Use the
    eigenvalue solver in scipy.linalg to calculate the steady state
    of the PageRank algorithm
    Inputs:
    adj - A NumPy array representing the adjacency matrix of a directed graph
    N - Restrict the computation to the first `N` nodes of the graph.
            Defaults to N=None; in this case, the entire matrix is used.
    d     - The damping factor, a float between 0 and 1.
            Defaults to .85.
    Returns:
    The approximation to the steady state.
    '''
    if N is None:
        N = adj.shape[0]

    E = np.ones((N,N))
    K = calculateK(adj,N)[:N,:N]
    B = d*K + (1.-d)/N*E
    eig_vec = la.eig(B)[1][:,0].real
    return eig_vec/np.sum(eig_vec)
    
def problem5(filename='ncaa2013.csv'):
    '''
    Create an adjacency matrix from the input file.
    Using iter_solve with d = 0.7, run the PageRank algorithm on the adjacency 
    matrix to estimate the rankings of the teams.
    Inputs:
    filename - Name of a .txt file containing data for basketball games. 
        Should contain a header row: 'Winning team,Losing team",
        after which each row contains the names of two teams,
        winning and losing, separated by a comma
    Returns:
    sorted_ranks - The array of ranks output by iter_solve, sorted from highest
        to lowest.
    sorted_teams - List of team names, sorted from highest rank to lowest rank.   
    '''

    teamSet = set()
    with open(filename) as inFile:
        inFile.readline()
        startIndex = 0
        for line in inFile:
            a = line.strip().split(',')
            teamSet.add(a[0])
            teamSet.add(a[1])
        teamList = list(teamSet)
        numberOfTeams = len(teamList)
        index = [x for x in xrange(numberOfTeams)]
        adj = np.zeros((numberOfTeams, numberOfTeams))
        teamDict = dict(zip(teamList, index))
        inFile.seek(0,0)
        inFile.readline()
        for line in inFile:
            a = line.strip().split(',')
            adj[teamDict[a[1]], teamDict[a[0]]] = 1

    steadyState = iter_solve(adj, d=.7)
    sorted_ranks = np.argsort(steadyState)[::-1]
    sorted_teams = [teamList[i] for i in sorted_ranks.flatten()]
    return sorted_ranks, sorted_teams[::-1]
    
def problem6():
    '''
    Optional problem: Load in and explore any one of the SNAP datasets.
    Run the PageRank algorithm on the adjacency matrix.
    If you can, create sparse versions of your algorithms from the previous
    problems so that they can handle more nodes.
    '''
    pass
