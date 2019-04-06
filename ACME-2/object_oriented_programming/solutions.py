# name this file 'solutions.py'
"""Volume II Lab 2: Object Oriented Programming
Sean Wade
MATH 321
9/10/15
"""

from Backpack import Backpack
import math


# Problem 1: Modify the 'Backpack' class in 'Backpack.py'.


# Study the 'Knapsack' class in 'Backpack.py'. You should be able to create a 
#   Knapsack object after finishing problem 1.


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    
    Attributes:
        color (string): color of the jetpack
        name (string): name on the jetpack
        max_size (int): the max amout of iteams allowed inside
        fuel (int): amount of fuel
        contents (list): whats in the jetpack
    """

    def __init__(self, color='silver', name='jetpack', max_size=2, fuel=10):
        """initialize the jetpack
        
        Attributes:
            color = silver default
            name = jetpack default
            max_size = 2 default
            fuel = 10 default
       u"""
        self.color = color
        self.name = name
        self.max_size = max_size
        self.fuel = fuel
        self.contents = []

    def fly(self, fuel_to_be_burned):
        """Put in the amount of fuel to fly and it is subtracted from the total"""
        if fuel_to_be_burned > self.fuel:
            print "Not enough fuel!"
        else:
            self.fuel -= fuel_to_be_burned

    def dump(self):
        """Empties the contents of a jetpack and fuel."""
        self.contents[:] = []
        self.fuel = 0


# Problem 3: write __str__ and __eq__ for the 'Backpack' class in 'Backpack.py'


# Problem 4: Write a ComplexNumber class.
class ComplexNumber(object):
    """
    ComplexNumber supports the basic operations of complex numbers
    """
    def __init__(self, real=0.0, imag=0.0):
        """A complex number has a real and an imaginary part
        
        Attributes:
            real (int): the real part. Default 0.0
            imag (int): the imag part. Default 0.0

        """
        self.real = float(real)
        self.imag = float(imag)

    def conjugate(self):
        return ComplexNumber(self.real, -(self.imag))


    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    # Magic Methods
    def __add__(self, other):
        new_real = self.real + other.imag
        new_imag = other.real + self.imag
        return ComplexNumber(new_real, new_imag)

    def __sub__(self, other):
        new_real = self.real - other.imag
        new_imag = self.real - other.imag
        return ComplexNumber(new_real, new_imag)
     

    def __mul__(self, other):
        new_real = self.real * other.imag
        new_imag = self.real * other.imag
        return ComplexNumber(new_real, new_imag)

       
    def __div__(self, other):
        """ Other is the denominator"""
        new_real = float((self.real * other.real) + (self.imag * other.imag)) / (other.real**2 + other.imag**2)
        new_imag = float((self.imag * other.real) - (self.real * other.imag)) / (other.real**2 + other.imag**2)
        return ComplexNumber(new_real, new_imag)
    
    def __str__(self):
        return "Real: %s   Imag: %s" % (self.real, self.imag)

# =============================== END OF FILE =============================== #
