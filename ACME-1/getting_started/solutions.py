# solutions.py
"""Volume I Lab 1: Getting Started
Sean Wade
Math 345
"""


# Problem 1: Write and run a "Hello World" script.
def hello_world():
    print "Hello World"


# Problem 2: Implement this function.
def sphere_volume(r):
    """Return the volume of the sphere of radius 'r'."""
    
    vol = (4.0)/3*(3.14159)*r**3
    return vol


# Problem 3: Implement the first_half() and reverse() functions.
def first_half(my_string):
    """Return the first half of the string 'my_string'.

    Example:
        >>> first_half("python")
        'pyt'
    """
    str_size = len(my_string)
    str_size = str_size / 2
    return my_string[0:str_size]

def reverse(my_string):
    """Return the reverse of the string 'my_string'.
    
    Example:
        >>> reverse("python")
        'nohtyp'
    """

    return my_string[::-1]

    
# Problem 4: Perform list operations
# For the grader, do not change the name of 'my_list'.
my_list = ["ant", "baboon", "cat", "dog"]

# Put your code here
# ------------------
# 1
my_list.append("elephant")
# 2
my_list.remove("ant")
# 3
my_list.remove(my_list[1])
# 4
my_list[2] = "donkey"
# 5
my_list.append("fox")


    
# Problem 5: Implement this function.
def pig_latin(word):
    """Translate the string 'word' into Pig Latin
    
    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = set('aeiouAEIOU')
    if word[0] in vowels:
        word += "hay"
    else:
        firstLetter = word[0]
        word = word[1:]
        word += firstLetter + "ay"
    return word

        
# Problem 6: Implement this function.
def int_to_string(my_list):
    """Use a dictionary to translate a list of numbers 1-26 to corresponding
    lowercase letters of the alphabet. 1 -> a, 2 -> b, 3 -> c, and so on.
    
    Example:
        >>> int_to_string([13, 1, 20, 8])
        ['m', 'a', 't', 'h'] 
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabetDictionary = dict((alphabet.index(item)+1, item) for item in alphabet)
    newWord = []
    for num in my_list:
        newWord.append(alphabetDictionary[num])
    return newWord


# Problem 7: Implement this generator.
def squares(n):
    """Yield all squares less than 'n'.

    Example:
        >>> for i in squares(10):
        ...     print(i)
        ... 
        0
        1
        4
        9
    """
    count = 0
    while count**2 < n:
        yield count**2
        count += 1


# Problem 8: Implement this function.
def stringify(my_list):
    """Using a list comprehension, convert the list of integers 'my_list'
    to a list of strings. Return the new list.

    Example:
        >>> stringify([1, 2, 3])
        ['1', '2', '3']
    """
    return [str(x) for x in my_list]


# Problem 9: Implement this function and use it to approximate ln(2).
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximae ln(2).
    """
    harmonicSeries = [(-1)**(x+1) * 1.0/x for x in range(1, n+1)]
    curSum = 0
    for place in harmonicSeries:
        curSum += place
    return curSum


ln2 = .69314
