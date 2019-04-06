import math

class ComplexNumber:
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

def test():
    a = ComplexNumber(4,5)
    b = ComplexNumber(2,2)
    print a.conjugate()
    print a.norm()
    print a.imag
    print a / b 
    print a * b
    print a - b
    print a + b


test()
