# LinkedLists.py
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Auxiliary file. Modify this file for problems 1-5.
Sean Wade
MATH 322
09.21.15
"""

# Problem 1: Add the magic methods __str__, __lt__, __eq__, and __gt__.
class Node(object):
    """A Node class for storing data."""
    def __init__(self, data):
        """Construct a new node that stores some data."""
        self.data = data

    def __str__(self):
        return str(self.data)

    def __lt__(self, other):
        return self.data < other

    def __eq__(self, other):
        return self.data == other

    def __gt__(self, other):
        return self.data > other


class LinkedListNode(Node):
    """A Node class for linked lists. Inherits from the 'Node' class.
    Contains a reference to the next node in the list.
    """
    def __init__(self, data):
        """Construct a Node and initialize an attribute for
        the next node in the list.
        """
        Node.__init__(self, data)
        self.next = None

# Problems 2-4: Finish implementing this class.
class LinkedList(object):
    """Singly-linked list data structure class.
    The first node in the list is referenced to by 'head'.
    """
    def __init__(self):
        """Create a new empty linked list. Create the head
        attribute and set it to None since the list is empty.
        """
        self.head = None

    def add(self, data):
        """Create a new Node containing 'data' and add it to
        the end of the list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.head.data
            1
            >>> my_list.add(2)
            >>> my_list.head.next.data
            2
        """
        new_node = LinkedListNode(data)
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node
    
    # Problem 2: Implement the __str__ method so that a LinkedList instance can
    #   be printed out the same way that Python lists are printed.
    def __str__(self):
        """String representation: the same as a standard Python list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.add(2)
            >>> my_list.add(3)
            >>> print(my_list)
            [1, 2, 3]
            >>> str(my_list) == str([1,2,3])
            True
        """
        if self.head is None:
            return "[]"
        current_node = self.head
        str_list = []
        while current_node.next is not None:
            str_list.append(current_node.data)
            current_node = current_node.next
        if self.head != None:
            str_list.append(current_node.data)
        return str_list.__str__()

    # Problem 3: Finish implementing LinkedList.remove() so that if the node
    #   is not found, an exception is raised.
    def remove(self, data):
        """Remove the node containing 'data'. If the list is empty, or if the
        target node is not in the list, raise a ValueError with error message
        "<data> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.remove(2)
            >>> print(my_list)
            [1, 3]
            >>> my_list.remove(2)
            2 is not in the list.
            >>> print(my_list)
            [1, 3]
        """
        if self.head == None:
            raise ValueError("data is not in the list.")
        elif self.head.data == data: 
            self.head = self.head.next
        else:
            current_node = self.head
            while current_node.next != None and current_node.next.data != data:
                current_node = current_node.next
            if current_node.next == None:
                raise ValueError("data is not in the list.")
            new_next_node = current_node.next.next
            current_node.next = new_next_node

    # Problem 4: Implement LinkedList.insert().
    def insert(self, data, place):
        """Create a new Node containing 'data'. Insert it into the list before
        the first Node in the list containing 'place'. If the list is empty, or
        if there is no node containing 'place' in the list, raise a ValueError
        with error message "<place> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 3]
            >>> my_list.insert(2,3)
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.insert(2,4)
            4 is not in the list.
        """
        if self.head == None:
            raise ValueError("%s is not in the list." % place)
        elif self.head.data == place:
            new_head = LinkedListNode(data)
            new_head.next = self.head
            self.head = new_head
        else:
            current_node = self.head
            while current_node.next != None and current_node.next.data != place:
                current_node = current_node.next
            if current_node.next == None:
                raise ValueError("%s is not in the list." % place)
            temp_pointer = current_node.next
            current_node.next = LinkedListNode(data)
            current_node.next.next = temp_pointer







class DoublyLinkedListNode(LinkedListNode):
    """A Node class for doubly-linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the list.
    """
    def __init__(self,data):
        """Initialize the next and prev attributes."""
        Node.__init__(self,data)
        self.next = None
        self.prev = None


# Problem 5: Implement this class.
class DoublyLinkedList(LinkedList):
    """Doubly-linked list data structure class. Inherits from the 'LinkedList'
    class. Has a 'head' for the front of the list and a 'tail' for the end.
    """
    def __init__(self):
        LinkedList.__init__(self)
        self.tail = None

    def add(self, data):
        """Create a new Node containing 'data' and add it to
        the end of the list.

        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.head.data
            1
            >>> my_list.add(2)
            >>> my_list.head.next.data
            2
        """
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node
            self.tail = current_node.next

    def remove(self, data):
        """Remove the node containing 'data'. If the list is empty, or if the
        target node is not in the list, raise a ValueError with error message
        "<data> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.remove(2)
            >>> print(my_list)
            [1, 3]
            >>> my_list.remove(2)
            2 is not in the list.
            >>> print(my_list)
            [1, 3]
        """
        if self.head == None:
            raise ValueError("data is not in the list.")
        elif self.head.data == data:
            if self.head == self.tail:
                self.tail = None
            self.head = self.head.next 
        else:
            current_node = self.head
            while current_node.next != None and current_node.next.data != data:
                current_node = current_node.next
            if current_node.next == None:
                raise ValueError("data is not in the list.")
            new_next_node = current_node.next.next
            if self.tail == data:
                self.tail = new_next_node
            current_node.next = new_next_node

    def insert(self, data, place):
        """Create a new Node containing 'data'. Insert it into the list before
        the first Node in the list containing 'place'. If the list is empty, or
        if there is no node containing 'place' in the list, raise a ValueError
        with error message "<place> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 3]
            >>> my_list.insert(2,3)
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.insert(2,4)
            4 is not in the list.
        """
        if self.head == None:
            raise ValueError("%s is not in the list." % place)
        elif self.head.data == place:
            if self.head == self.tail:
                self.tail = None
            new_head = LinkedListNode(data)
            new_head.next = self.head
            self.head = new_head
        else:
            current_node = self.head
            while current_node.next != None and current_node.next.data != place:
                current_node = current_node.next
            if current_node.next == None:
                raise ValueError("%s is not in the list." % place)
            temp_pointer = current_node.next
            current_node.next = LinkedListNode(data)
            current_node.next.next = temp_pointer


def linked_list_test():
    a = LinkedList()
    a.add("a")
    a.add("c")
    a.add("f")
    a.add("h")
    print a


def double_linked_list_test():
    a = DoublyLinkedList()
    a.add("a")
    a.add("b")
    a.add("c")
    a.add("d")
    a.insert(1,"d")
    print a

# Problem 6: Implement this class. Use an instance of your object to implement
# the sort_words() function in solutions.py.
class SortedLinkedList(DoublyLinkedList):
    """Sorted doubly-linked list data structure class."""

    # Overload add() and insert().
    def add(self, data):
        """Create a new Node containing 'data' and insert it at the
        appropriate location to preserve list sorting.
        
        Example:
            >>> print(my_list)
            [3, 5]
            >>> my_list.add(2)
            >>> my_list.add(4)
            >>> my_list.add(6)
            >>> print(my_list)
            [2, 3, 4, 5, 6]
        """

    def add(self, data):
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        elif self.head.data >= data:
            new_node.next = self.head
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next != None and current_node.next.data <= data:
                current_node = current_node.next
            if current_node.next == None and current_node.data > data:
                current_node.next = new_node
                self.tail = new_node
            else:
                temp = current_node.next
                current_node.next = new_node
                new_node.next = temp


    def insert(self, *args):
        raise ValueError("insert() has been disabled for this class.")


class Deque(DoublyLinkedList):
    """Sean Wade's implementation of a deque."""


    def add(self, *args):
        raise ValueError("add() has been disabled for this class.")
    def remove(self, *args):
        raise ValueError("remove() has been disabled for this class.")
    def insert(self, *args):
        raise ValueError("insert() has been disabled for this class.")
    

    def appendleft(self, data):
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node

    def popleft(self):
        if self.head == None:
            raise ValueError("it is empty.")
        else:
            data_to_return = self.head.data
            self.head = self.head.next 
            return data_to_return
        
    def append(self, data):
        new_node = DoublyLinkedListNode(data)
        if self.tail is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def pop(self):
        if self.tail == None:
            raise ValueError("it is empty.")
        elif self.head.next is None:
            to_return = self.head.data
            self.head = None
            self.tail = None
            return to_return

        else:
            current_node = self.head
            while current_node.next.next is not None:
                current_node = current_node.next
            to_return = current_node.next.data
            self.tail = current_node
            self.tail.next = None
            return to_return


class Stack(Deque):
    """Sean Wade's implementation of a stack."""

    def appendleft(self, *args):
        raise ValueError("appendleft() has been disabled for this class.")

    def popleft(self, *args):
        raise ValueError("popleft() has been disabled for this class.")

    def append(self, *args):
        raise ValueError("append() has been disabled for this class.")


    def push(self, data):
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node
            self.tail = current_node.next


class Queue(Deque):
    """Sean Wade's implementation of a queue."""

    def appendleft(self, *args):
        raise ValueError("appendleft() has been disabled for this class.")

    def popleft(self, *args):
        raise ValueError("popleft() has been disabled for this class.")

    def append(self, *args):
        raise ValueError("append() has been disabled for this class.")


    def push(self, data):
        """Pushes onto the back of the queue."""
        new_node = DoublyLinkedListNode(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def pop(self):
        """Pop of the front of the queue."""
        if self.head is None:
            raise ValueError("it is empty.")
        else:
            to_return = self.head.data
            self.head = self.head.next
            return to_return

# =============================== END OF FILE =============================== #
