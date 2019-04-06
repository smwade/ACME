# Name this file 'solutions.py'.
"""Volume II Lab 22: Dynamic Optimization (Value Function Iteration).
Sean Wade
Value Function Iteration
"""
import numpy as np
import scipy as sp
from scipy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def eatCake(beta, N, W_max=1, T=None, finite=True, plot=False):
    """Solve the finite- or infinite-horizon cake-eating problem using
    Value Function iteration.
    
    Inputs:
        beta (float): Discount factor.
        N (int): The number of discrete cake values.
        W_max (int): The original size of the cake.
        T (int): The final time period. Defaults to None.
        finite (bool): If True, solve the finite-horizon problem. If False,
            solve the infinite-horizon problem.
        plot (bool): If True, plot the value function surface and policy
            function.

    Returns:
        values ((N, T+2) ndarray if finite=True, (N,) ndarray if finite=False):
            The value function at each time period for each state (this is also
            called V in the lab).
        psi ((N, T+1) ndarray if finite=True, (N,) ndarray if finite=False):
            The policy at each time period for each state.
    """
    w = np.linspace(0, W_max, N)
    u = np.zeros((N,N))

    for i in range(N):
        u[i] = w[i] * np.ones(N) - w
        u[i][:i+1] = np.sqrt(u[i][:i+1])
        u[i][i+1:] = -1e10


    if finite:
        psi = np.zeros((N, T+1))
        V = np.zeros((N,T+2))
        for t in xrange(T,-1,-1):
            temp = (u.T + beta * V[:,t+1]).T
            V[:,t] = np.max(temp, axis=1)
            best = np.argmax(temp, axis=1)
            psi[:,t] = w[best]
        if plot:
            W = np.linspace(0, W_max, N)
            x = np.arange(0, N)
            y = np.arange(0, T+2)
            X, Y = np.meshgrid(x, y)
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            ax1.plot_surface(W[X], Y, np.transpose(V), cmap=cm.coolwarm)
            plt.show()
            fig2 = plt.figure()
            ax2 = Axes3D(fig2)
            y = np.arange(0,T+1)
            X, Y = np.meshgrid(x, y)
            ax2.plot_surface(W[X], Y, np.transpose(psi), cmap=cm.coolwarm)
            plt.show()
    else:
        V = np.zeros(N)
        delta = 1
        while delta > 1e-9:
            V_old = np.copy(V)
            temp = (u.T + beta * V_old).T
            # maybe one
            V = np.max(temp, axis=0)
            best = np.argmax(temp, axis=0)
            delta = la.norm(V - V_old)
        psi = w[best]
        if plot:
            W = np.linspace(0, W_max, N)
            x = np.arange(0, N)
            y = np.arange(0, T+2)
            X, Y = np.meshgrid(x, y)
            fig1 = plt.figure()
            ax1 = Axes3D(fig1)
            ax1.plot_surface(W[X], Y, np.transpose(V), cmap=cm.coolwarm)
            plt.show()
            fig2 = plt.figure()
            ax2 = Axes3D(fig2)
            y = np.arange(0,T+1)
            X, Y = np.meshgrid(x, y)
            ax2.plot_surface(W[X], Y, np.transpose(psi), cmap=cm.coolwarm)
            plt.show()    

    return V, psi


def prob2():
    """Call eatCake() with the parameters specified in the lab."""
    eatCake(beta=.9, N=100, T=1000, plot=True)


def prob3():
    """Modify eatCake() to deal with the infinite case.
    Call eatCake() with the parameters specified in part 6 of the problem.
    """
    eatCake(beta=.9, N=100, T=1000, plot=True, finite=False)
