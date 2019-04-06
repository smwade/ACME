# spec.py
"""Volume 1, Lab 16: Importance Sampling and Monte Carlo Simulations.
Sean Wade
Math 347
1/24/2016
"""

from __future__ import division
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

# Problem 1 
def prob1(n):
    """Approximate the probability that a random draw from the standard
    normal distribution will be greater than 3.
    Returns: your estimated probability.
    """
    random_samples = np.random.randn(n)
    h = lambda x: x > 3
    estimator = (1 / n) * np.sum(h(random_samples))
    return estimator

# Problem 2
def prob2():
    """Answer the following question using importance sampling: 
            A tech support hotline receives an average of 2 calls per 
            minute. What is the probability that they will have to wait 
            at least 10 minutes to receive 9 calls?
    Returns:
        IS (array) - an array of estimates using 
            [5000, 10000, 15000, ..., 500000] as number of 
            sample points."""
    def importance(n):
        h = lambda x: x>10
        f = lambda x: stats.gamma.pdf(x, 9, scale=.5)
        g = lambda x: stats.norm(loc=10, scale=1).pdf(x)

        X = np.random.normal(loc=10,scale=1,size=n)
        return 1./n * np.sum(h(X)*f(X)/g(X))

    npts = np.arange(5000, 500001, 500)
    return np.vectorize(importance)(npts)

# Problem 3
def prob3():
    """Plot the errors of Monte Carlo Simulation vs Importance Sampling
    for the prob2()."""
    xpts = np.arange(5000, 500001, 500)
    ypts = prob2()

    h = lambda x : x > 10
    MC_estimates = []
    for N in xpts:
        X = np.random.gamma(9,scale=0.5,size=N)
        MC = 1./N*np.sum(h(X))
        MC_estimates.append(MC)
    MC_estimates = np.array(MC_estimates)

    my_error = np.abs(ypts - (1 - stats.gamma(a=9,scale=.5).cdf(10)))
    their_error = np.abs(MC_estimates - (1 - stats.gamma(a=9,scale=.5).cdf(10)))
    plt.plot(xpts, my_error, label="Importance Sampling")
    plt.plot(xpts, their_error, label="Monte Carlo")
    plt.legend(loc=0)
    plt.show()

# Problem 4
def prob4():
    """Approximate the probability that a random draw from the
    multivariate standard normal distribution will be less than -1 in 
    the x-direction and greater than 1 in the y-direction.
    Returns: your estimated probability"""
    f =  lambda x: stats.multivariate_normal(mean=np.array([0,0])).pdf(x)
    h = lambda x: x[0] < -1 and x[1] > 1
    g = lambda x: stats.multivariate_normal(mean=np.array([-1,1])).pdf(x)

    N = 10000
    X = np.random.normal(loc=-1, scale=1, size=N)
    Y = np.random.normal(loc=1, scale=1, size=N)
    pt = np.vstack((X,Y)).T
    final = 0
    for i in xrange(N):
        final += (1./N) * np.sum(h(pt[i])*f(pt[i])/g(pt[i]))

    return final
print prob4()
