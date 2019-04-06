"""Volume II Lab 19: Trust Region Methods
Sean Wade
"""

import numpy as np
from scipy import linalg as la
from scipy import optimize as op


# Problem 1
def trustRegion(f,grad,hess,subprob,x0,r0,rmax=2.,eta=1./16,gtol=1e-5):
    """Implement the trust regions method.
    
    Parameters:
        f (function): The objective function to minimize.
        g (function): The gradient (or approximate gradient) of the objective
            function 'f'.
        hess (function): The hessian (or approximate hessian) of the objective
            function 'f'.
        subprob (function): Returns the step p_k.
        x0 (ndarray of shape (n,)): The initial point.
        r0 (float): The initial trust-region radius.
        rmax (float): The max value for trust-region radii.
        eta (float in [0,0.25)): Acceptance threshold.
        gtol (float): Convergence threshold.
        
    
    Returns:
        x (ndarray): the minimizer of f.
    
    Notes:
        The functions 'f', 'g', and 'hess' should all take a single parameter.
        The function 'subprob' takes as parameters a gradient vector, hessian
            matrix, and radius.
    """

    while la.norm(grad(x0)) > gtol:
        pk = subprob(grad(x0), hess(x0), r0)
        mk = np.dot(pk, grad(x0)) + .5 * np.dot(np.dot(pk, hess(x0)), pk)
        rho = -(f(x0) - f(x0 + pk)) / mk

        if rho < .25:
            r0 = .25 * r0
        else:
            if rho > .075 and la.norm(pk) == r0:
                r0 = min(2*r0, rmax)
            else:
                r0 = r0
        if rho > eta:
            x0 += pk
        else:
            x0 = x0
    return x0

# Problem 2   
def dogleg(gk,Hk,rk):
    """Calculate the dogleg minimizer of the quadratic model function.
    
    Parameters:
        gk (ndarray of shape (n,)): The current gradient of the objective
            function.
        Hk (ndarray of shape (n,n)): The current (or approximate) hessian.
        rk (float): The current trust region radius
    
    Returns:
        pk (ndarray of shape (n,)): The dogleg minimizer of the model function.
    """

    pb = la.solve(-Hk, gk)
    pu = -np.dot(np.dot(gk, gk) / np.dot(np.dot(gk, Hk), gk), gk)

    a = np.dot(pb, pb) - 2 * np.dot(pb, pu) + np.dot(pu, pu)
    b = 2*np.dot(pb, pu) - 2*np.dot(pu, pu)
    c = np.dot(pu, pu) - rk*rk

    tao = np.roots([a, b, c])

    if la.norm(pb) <= rk:
        pk = pb

    elif la.norm(pu) >= rk:
        pk = np.dot(rk, pu) / la.norm(pu)

    else:
        pk = pu + np.max(tao) * (pb - pu)

    return pk


# Problem 3
def problem3():
    """Test your trustRegion() method on the Rosenbrock function.
    Define x0 = np.array([10.,10.]) and r = .25
    Return the minimizer.
    """

    x = np.array([10.,10])
    rmax=2.
    r=.25
    eta=1./16
    tol=1e-5
    opts = {'initial_trust_radius':r, 'max_trust_radius':rmax, 'eta':eta, 'gtol':tol}
    sol1 = op.minimize(op.rosen, x, method='dogleg', jac=op.rosen_der, hess=op.rosen_hess, options=opts)
    sol2 = trustRegion(op.rosen, op.rosen_der, op.rosen_hess, dogleg, x, r, rmax, eta, gtol=tol)
    return sol2

# Problem 4
def problem4():
    """Solve the described non-linear system of equations.
    Return the minimizer.
    """

    def r(x):
        return np.array([np.sin(x[0])*np.cos(x[1]) - 4*np.cos(x[0])*np.sin(x[1]),
                        np.sin(x[1])*np.cos(x[0]) - 4*np.cos(x[1])*np.sin(x[0])])

    def f(x):
        return .5*(r(x)**2).sum()

    def J(x):
        return np.array([[np.cos(x[0])*np.cos(x[1]) + 4*np.sin(x[0])*np.sin(x[1]),
                        -np.sin(x[0])*np.sin(x[1]) - 4*np.cos(x[0])*np.cos(x[1])],
                        [np.sin(x[1])*np.sin(x[0]) - 4*np.cos(x[1])*np.cos(x[0]),
                        -np.cos(x[1])*np.cos(x[0]) + 4*np.sin(x[1])*np.sin(x[0])]])

    def g(x):
        return J(x).dot(r(x))

    def H(x):
        return J(x).T.dot(J(x))

    tol=1e-5
    maxr=2.
    rr=.25
    eta=1./16
    
    x = np.array([3.5, -2.5])
    xstar = trustRegion(f,g,H,dogleg,x,rr,maxr,eta=eta,gtol=tol)

    return xstar
