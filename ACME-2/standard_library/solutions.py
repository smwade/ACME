
"""Volume II Lab 1: The Standard Library
Sean Wade
321
9/3/15
"""

# Add import statements here.
import numpy as np
import calculator as calc
import sys
import matrix_multiply
import time

# In future labs, do not modify any PROVIDED import statements.
# You may always add others as needed.


# Problem 1: Implement this function.
def prob1(l):
    """Accept a list 'l' of numbers as input and return a new list with the
    minimum, maximum, and average of the contents of 'l'.
    """
    answerList = []
    minimum = min(l)
    maximum = max(l)
    average = np.mean(l)

    answerList.append(minimum)
    answerList.append(maximum)
    answerList.append(average)
    return answerList


# Problem 2: Implement this function.
def prob2():
    """Determine which Python objects are mutable and which are immutable. Test
    numbers, strings, lists, tuples, and dictionaries. Print your results to the
    terminal using the print() function.
   """
    # for numbers
    num1 = 4
    num2 = num1
    num1 = 3
    print(num1, num2)
        
    num1 = 4
    num2 = num1
    num1 += 3
    print(num1, num2)

    # for strings
    str1 = "cat"
    str2 = str1
    str1 = "dog"
    print(str1, str2)

    str1 = "cat"
    str2 = str1
    str1 += 'a'
    print(str1, str2)

    # for lists
    list1 = [1, 2, 3]
    list2 = list1
    list1.append(4)
    print(list1, list2)

    # for tuples
    tup1 = (1, 2, 3)
    tup2 = tup1
    tup1 += (4,)
    print(tup1, tup2)

    # for dictionaries
    dic1 = {1: 'a', 2: 'b', 3: 'c'}
    dic2 = dic1
    dic1[4] = 'd'
    print(dic1, dic2)
    
    # interpret the results
    print("==============================================")
    print("numbers are: immutable")
    print("strings are: immutable")
    print("lists are: mutable")
    print("tupples are: immutable")
    print("dicitonaries are: mutable")

# Problem 3: Create a 'calculator' module and use it to implement this function.
def prob3(a,b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any methods other than those that are imported from the
    'calculator' module.
    
    Parameters:
        a (float): the length one of the sides of the triangle.
        b (float): the length the other nonhypotenuse side of the triangle.
    
    Returns:
        The length of the triangle's hypotenuse.
    """
    c =  calc.squarRoot(calc.add(calc.mult(a,a),calc.mult(b,b)))
    return c
   

# Problem 4: Utilize the 'matrix_multiply' module and 'matrices.npz' file to
#   implement this function.
def prob4():
    """If no command line argument is given, print "No Input."
    If anything other than "matrices.npz is given, print "Incorrect Input."
    If "matrices.npz" is given as a command line argument, use functions
    from the provided 'matrix_multiply' module to load two matrices, then
    time how long each method takes to multiply the two matrices together.
    Print your results to the terminal.
    """
    if len(sys.argv) < 2:
        print("No Input.")
    elif sys.argv[1] != "matrices.npz":
        print("Incorrect Input.")
    else:
        A,B = matrix_multiply.load_matrices("matrices.npz")
        
        start = time.time()
        matrix_multiply.method1(A,B)
        end = time.time()
        method1Time = end - start

        start = time.time()
        matrix_multiply.method2(A,B)
        end = time.time()
        method2Time = end - start

        start = time.time()
        matrix_multiply.method3(A,B)
        end = time.time()
        method3Time = end - start

        print("time for method1 is: " + str(method1Time))
        print("time for method2 is: " + str(method2Time))
        print("time for method3 is: " + str(method3Time))
# Everything under this 'if' statement is executed when this file is run from
#   the terminal. In this case, if we enter 'python solutions.py word' into
#   the terminal, then sys.argv is ['solutions.py', 'word'], and prob4() is
#   executed. Note that the arguments are parsed as strings. Do not modify.
if __name__ == "__main__":
    prob4()


# ============================== END OF FILE ================================ #
