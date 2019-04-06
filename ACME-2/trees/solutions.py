"""Volume II Lab 5: Data Structures II (Trees)
Sean Wade
MATH 321
9/30/16
"""

from Trees import BST
from Trees import AVL
import WordList
from LinkedLists import LinkedList
import numpy as np
import timeit
from matplotlib import pyplot as plt
import random


def iterative_search(linkedlist, data):
    """Find the node containing 'data' using an iterative approach.
    If there is no such node in the list, or if the list is empty,
    raise a ValueError with error message "<data> is not in the list."
    
    Inputs:
        linkedlist (LinkedList): a linked list object
        data: the data to find in the list.
    
    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    # Start the search at the head.
    current = linkedlist.head
    
    # Iterate through the list, checking the data of each node.
    while current is not None:
        if current.data == data:
            return current
        current = current.next
    
    # If 'current' no longer points to a Node, raise a value error.
    raise ValueError(str(data) + " is not in the list.")


# Problem 1: rewrite iterative_search() using recursion.
def recursive_search(linkedlist, data):
    """Find the node containing 'data' using a recursive approach.
    If there is no such node in the list, raise a ValueError with error
    message "<data> is not in the list."
    
    Inputs:
        linkedlist (LinkedList): a linked list object
        data: the data to find in the list.
    
    Returns:
        The node in 'linkedlist' containing 'data'.
    """
    cur_node = linkedlist.head
    def inner(cur_node):
        if cur_node is None:
            raise ValueError(str(data) + " is not in the list.")
        elif cur_node.data == data:
            return cur_node
        else:
            return inner(cur_node.next)
    return inner(cur_node)



# Problem 2: Implement BST.insert() in Trees.py.


# Problem 3: Implement BST.remove() in Trees.py


# Problem 4: Test build and search speeds for LinkedList, BST, and AVL objects.
def plot_times(filename="English.txt", start=500, stop=5500, step=500):
    """Vary n from 'start' to 'stop', incrementing by 'step'. At each
    iteration, use the create_word_list() from the 'WordList' module to
    generate a list of n randomized words from the specified file.
    
    Time (separately) how long it takes to load a LinkedList, a BST, and
    an AVL with the data set.
    
    Choose 5 random words from the data set. Time how long it takes to
    find each word in each object. Calculate the average search time for
    each object.
    
    Create one plot with two subplots. In the first subplot, plot the
    number of words in each dataset against the build time for each object.
    In the second subplot, plot the number of words against the search time
    for each object.
    
    Inputs:
        filename (str): the file to use in creating the data sets.
        start (int): the lower bound on the sample interval.
        stop (int): the upper bound on the sample interval.
        step (int): the space between points in the sample interval.
    
    Returns:
        Show the plot, but do not return any values.
    """

    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    def add_all(A, my_list):
        for x in my_list:
            A.add(x)

    def add_all_tree(A, my_list):
        for x in my_list:
            A.insert(x)

    def find_it(A, to_find):
        A.find(to_find)

    def find_average(A, my_list):
        find_times = []
        for x in range(5):
            to_find = random.choice(my_list)
            # to_find = my_list[x]
            wrapped = wrapper(find_it, A, to_find)
            find_times.append(timeit.timeit(wrapped, number=1))
        return np.mean(find_times)





    word_list = WordList.create_word_list()
    word_list = np.random.permutation(word_list)
    x_values = range(start, stop, step)
    list_times = []
    bst_times = []
    avl_times = []
    find_list= []
    find_bst= []
    find_avl= []
    A = LinkedList()
    B = BST()
    C = AVL()

    for x in x_values:
        wrapped = wrapper(add_all, A, word_list[:int(x)])
        list_times.append(timeit.timeit(wrapped, number=1))
        find_list.append(find_average(A, word_list[:int(x)]))
        A.clear()


    for x in x_values:
        wrapped = wrapper(add_all_tree, B, word_list[:int(x)])
        bst_times.append(timeit.timeit(wrapped, number=1))
        find_bst.append(find_average(B, word_list[:int(x)]))
        B.clear()

    for x in x_values:
        wrapped = wrapper(add_all_tree, C, word_list[:int(x)])
        avl_times.append(timeit.timeit(wrapped, number=1))
        find_avl.append(find_average(C, word_list[:int(x)]))
        C.clear()




    plt.subplot(121)
    plt.plot(x_values, list_times, label='Linked List')
    plt.plot(x_values, bst_times, label='BST')
    plt.plot(x_values, avl_times, label='AVL')
    plt.legend(loc='upper left')
    plt.xlabel('data points')
    plt.ylabel('seconds')

    plt.subplot(122)
    plt.plot(x_values, find_list,label='Linked List')
    plt.plot(x_values, find_bst, label='BST')
    plt.plot(x_values, find_avl, label='AVL')
    plt.legend(loc='upper left')
    plt.xlabel('data points')
    plt.ylabel('seconds')

    plt.show()

    plt.xlabel('data points')

if __name__ == "__main__":
    A = BST()
    A.insert(2)
    A.insert(1)
    A.insert(7)
    A.insert(6)
    A.insert(5)
    A.insert(4)
    A.insert(3)
    print A
    A.remove(2)
    print A
    A.remove(3)
    print A


# =============================== END OF FILE =============================== #
