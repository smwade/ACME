"""Volume I Lab 3: Plotting with matplotlib
Sean Wade
9/15/15
"""

# Add your import statements here.
from matplotlib import pyplot as plt
import numpy as np
from mayavi import mlab



# Problem 1
def curve():
    """Plot the curve 1/(x-1) on [-2,6]. Plot the two sides of the curve separately
    (still with a single call to plt.plot()) so that the graph looks discontinuous 
    at x = 1.
    """
    x1 = np.linspace(-2, 1 - 10**(-4), 501)
    x2 = np.linspace(1 + 10**(-4), 6, 501)
    y1 = 1 / (x1-1)
    y2 = 1 / (x2-1)
    plt.plot(x1, y1, "m--", x2, y2, "m--", linewidth=4)
    plt.ylim(-6, 6)
    plt.show()




# Problem 2
def colormesh():
    """Plot the function f(x,y) = sin(x)sin(y)/(xy) on [-2*pi, 2*pi]x[-2*pi, 2*pi].
    Include the scale bar in your plot.
    """
    # neocomplete
    x = np.linspace(-2 * (np.pi), (2*np.pi), 401) 
    y = np.linspace(-2 * (np.pi), (2*np.pi), 401) 
    X, Y = np.meshgrid(x, y)
    f = (np.sin(X) * np.sin(Y)) / (X * Y) 
    plt.pcolormesh(X, Y, f, cmap='seismic')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.ylim((2* -np.pi, 2 * np.pi))
    plt.xlim((2 * -np.pi, 2* np.pi))
    plt.show()


# Problem 3
def histogram():
    """Plot a histogram and a scatter plot of 50 random numbers chosen in the
    interval [0,1)
    """
    x = np.random.rand(50)
    plt.subplot(1, 2, 1)
    plt.hist(x, bins=5, range=[.01, 1.0])
    plt.subplot(1, 2, 2)
    xcor = np.linspace(1,51,50)
    plt.scatter(xcor, x)
    mean = np.mean(x)
    y = np.linspace(mean, mean, 50)
    plt.plot(xcor, y, 'r')
    plt.xlim(1, 51)
    plt.show()

histogram()
   
# Problem 4
def ripple():
    """Plot z = sin(10(x^2 + y^2))/10 on [-1,1]x[-1,1] using Mayavi"""
    X, Y = np.mgrid[-1:1:0.025, -1:1:0.025]
    Z = np.sin(10 * (X**2 + Y**2))/10.0
    mlab.surf(X, Y, Z, colormap='RdYlGn')
    mlab.show()
