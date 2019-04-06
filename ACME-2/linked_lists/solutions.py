# name this file 'solutions.py'
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Sean Wade
Math 321
09/22/15
"""

# Do not modify the first two import statements (for the test driver)
from LinkedLists import LinkedListNode, LinkedList
from LinkedLists import DoublyLinkedList, SortedLinkedList
from WordList import create_word_list
from LinkedLists import Stack, Queue, Deque


# Problem 1: in LinkedLists.py, add magic methods to the Node class.

# Problems 2, 3, 4: Complete the implementation of the LinkedList class.

# Problem 5: Implement the DoublyLinkedList class.

# Problem 6: Implement the SortedLinkedList class in LinkedLists.py and the
# sort_words() function in this file.

# Conclude problem 6 by implementing this function.
def sort_words(filename = "English.txt"):
    """Use the 'create_word_list' method from the 'WordList' module to generate
    a scrambled list of words from the specified file. Use an instance of
    the SortedLinkedList class to sort the list. Then return the list.
    
    Inputs:
        filename (str, opt): the file to be parsed and sorted.
            Defaults to 'English.txt'.
    
    Returns:
        A SortedLinkedList object containing the sorted list of words.
    """
    my_list = create_word_list(filename)
    sorted_list = SortedLinkedList()
    for x in my_list:
        sorted_list.add(x)

    file = open("test.txt",'w')
    it = sorted_list.head
    out_list = []
    while it.next != None:
        file.write(it.data + '\n')
        it = it.next
    file.close()
    return sorted_list

# =========================== END OF File =========================== #
