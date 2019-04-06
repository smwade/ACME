"""Volume II Lab 15: Line Search Algorithms
Sean Wade
324
1/20/2016
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import seaborn
from scipy import linalg as la
from scipy.optimize import line_search, leastsq

# Problem 1
def newton1d(f, df, ddf, x, niter=10):
    """
    Perform Newton's method to minimize a function from R to R.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The first derivative of 'f'.
        ddf (function): The second derivative of 'f'.
        x (float): The initial guess.
        niter (int): The number of iterations. Defaults to 10.
    
    Returns:
        (float) The approximated minimizer.
    """
    x_next = x
    for _ in xrange(niter):
        x_cur = x_next
        x_next = x_cur - df(x_cur) / ddf(x_cur)
    return x_next
    

def test_newton():
    """Use the newton1d() function to minimixe f(x) = x^2 + sin(5x) with an
    initial guess of x_0 = 0. Also try other guesses farther away from the
    true minimizer, and note when the method fails to obtain the correct
    answer.

    Returns:
        (float) The true minimizer with an initial guess x_0 = 0.
        (float) The result of newton1d() with a bad initial guess.
    """
    f = lambda x: x**2 + np.sin(5*x)
    df = lambda x: 2*x + 5 * np.cos(5*x)
    ddf = lambda x: 2 - 25 * np.sin(5*x)
    return (newton1d(f, df, ddf, 0.), newton1d(f, df, ddf, 10.)) 


# Problem 2
def backtracking(f, slope, x, p, a=1, rho=.9, c=10e-4):
    """Perform a backtracking line search to satisfy the Armijo Conditions.

    Parameters:
        f (function): the twice-differentiable objective function.
        slope (float): The value of grad(f)^T p.
        x (ndarray of shape (n,)): The current iterate.
        p (ndarray of shape (n,)): The current search direction.
        a (float): The intial step length. (set to 1 in Newton and
            quasi-Newton methods)
        rho (float): A number in (0,1).
        c (float): A number in (0,1).
    
    Returns:
        (float) The computed step size satisfying the Armijo condition.
    """
    while f(x + a*p) > f(x) + c * a * slope:
        a = float(rho * a)
    return a


# Problem 3    
def gradientDescent(f, df, x, niter=10):
    """Minimize a function using gradient descent.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations to run.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """
    co = [x]
    for _ in xrange(niter):
        grad = df(x)
        slope = np.dot(grad, -grad)
        steps = backtracking(f, slope, x, -grad)
        x= x + (-grad) * steps
        co.append(x)
    return co

def newtonsMethod(f, df, ddf, x, niter=10):
    """Minimize a function using Newton's method.

    Parameters:
        f (function): The twice-differentiable objective function.
        df (function): The gradient of the function.
        ddf (function): The Hessian of the function.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (list of ndarrays) The sequence of points generated.
    """
    co = [x]
    for _ in xrange(niter):
        direction = np.dot(-la.inv(ddf(x)), df(x))
        slope = np.dot(direction, df(x))
        steps = backtracking(f, slope, x, direction)
        x = x + direction * steps
        co.append(x)
    return co

# Problem 4
def gaussNewton(f, df, jac, r, x, niter=10):
    """Solve a nonlinear least squares problem with Gauss-Newton method.

    Parameters:
        f (function): The objective function.
        df (function): The gradient of f.
        jac (function): The jacobian of the residual vector.
        r (function): The residual vector.
        x (ndarray of shape (n,)): The initial point.
        niter (int): The number of iterations.
    
    Returns:
        (ndarray of shape (n,)) The minimizer.
    """
    for _ in xrange(niter):
        p = la.solve(np.dot(jac(x).T,jac(x)), np.dot(-jac(x), r(x)))
        steps = line_search(f,df,x,p)
        x = x + steps * p
        k += 1
    return x



# Problem 5
def census():
    """Generate two plots: one that considers the first 8 decades of the US
    Census data (with the exponential model), and one that considers all 16
    decades of data (with the logistic model).
    """

    # Start with the first 8 decades of data.
    years1 = np.arange(8)
    pop1 = np.array([3.929,  5.308,  7.240,  9.638,
                    12.866, 17.069, 23.192, 31.443])

    # Now consider the first 16 decades.
    years2 = np.arange(16)
    pop2 = np.array([3.929,   5.308,   7.240,   9.638,
                    12.866,  17.069,  23.192,  31.443,
                    38.558,  50.156,  62.948,  75.996,
                    91.972, 105.711, 122.775, 131.669])
    # part 1 
    plt.scatter(years1, pop1)
    def f(x,t):
        return x[0] * np.exp(x[1] * (t + x[2]))

    r = lambda x: f(x,years1) - pop1

    x0 = np.array([150., .4, 2.5])
    x = leastsq(r, x0)[0]
    
    x_points = np.linspace(0,8,100)
    y = f(x, x_points)
    plt.plot(x_points, y, 'g')
    plt.show()
    
    plt.scatter(years2, pop2)
    def f2(x,t):
        return x[0] / (1 + np.exp(-x[1] * (t + x[2])))

    r2 = lambda x: f2(x,years2) - pop2
    x0 = np.array([150., .4, -15])
    x2 = leastsq(r2, x0)[0]
    
    x_points = np.linspace(0,16,100)
    y2 = f2(x2, x_points)
    plt.plot(x_points, y2, 'g')
    plt.show()
