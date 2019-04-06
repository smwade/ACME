# WordList.py
"""Volume II Lab 4: Data Structures 1 (Auxiliary file)
Use this module to complete problem 6. Do not modify.
"""

import numpy as np

# Use this function in problem 6 to implement sort_words().
def create_word_list(filename):
    """Read in a list of words from the specified file.
    Randomize the ordering and return the list.
    """
    myfile = open(filename, 'r')    # Open the file with read-only access
    contents = myfile.read()        # Read in the text from the file
    myfile.close()
    words = contents.split('\n')    # Get each word, separated by '\n'
    if words[-1] == "":
        words = words[:-1]          # Remove the last endline if needed
                                    # Randomize, convert to a list, and return.
    return list(np.random.permutation(words))

# You do not need this function, but read it anyway.
def export_word_list(wordList, outfile='Test.txt'):
    """Write a list of words to the specified file. You are not required
    to use this function, but it may be useful in testing sort_words().
    Note that 'words' must be a Python list, not a SortedLinkedList object.
    
    These two functions are examples of how file input / output works in
    Python. This concept will resurface many times in later labs.
    """
    myfile = open(outfile, 'w')     # Open the file with write-only access
    for w in wordList:              # Write each word to the file, with an
        myfile.write(w + '\n')      #   endline character after each word
    myfile.close()                  # Close the file.
