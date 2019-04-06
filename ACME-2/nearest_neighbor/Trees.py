# Trees.py
"""Volume II Lab 5: Data Structures II (Trees). Auxiliary File.
Contains Node and Tree classes. Modify this file for problems 2 and 3.
Sean Wade
MATH 321
10/1/15
"""


class BSTNode(object):
    """A Node class for Binary Search Trees. Contains some data, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the data attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.data = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.data < self.data
        self.right = None       # self.data < self.right.data
        
    def __str__(self):
        """String representation: the data contained in the node."""
        return str(self.data)

# Modify this class for problems 2 and 3.
class BST(object):
    """Binary Search Tree data structure class.
    The first node is referenced to by 'root'.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def clear(self):
        self.root = None
    
    def find(self, data):
        """Return the node containing 'data'. If there is no such node in the
        tree, raise a ValueError with error message "<data> is not in the tree."
        """
        # First, check to see if the tree is empty.
        if self.root is None:
            raise ValueError(str(data) + " is not in the tree.")
        
        # Define a recursive function to traverse the tree.
        def _step(current, item):
            """Recursively step through the tree until the node containing
            'item' is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end
                raise ValueError(str(data) + " is not in the tree.")
            if item == current.data:                # Base case 2: data matches
                return current
            if item < current.data:                 # Step to the left
                return _step(current.left,item)
            else:                                   # Step to the right
                return _step(current.right,item)
        
        # Start the recursion on the root of the tree.
        return _step(self.root, data)
    
    # Problem 2: Implement BST.insert().
    def insert(self, data):
        """Insert a new node containing 'data' at the appropriate location.
        Do not allow for duplicates in the tree: if there is already a node
        containing 'data' in the tree, raise a ValueError.
        
        Example:
            >>> b = BST()       |   >>> b.insert(1)     |       (4)
            >>> b.insert(4)     |   >>> print(b)        |       / \
            >>> b.insert(3)     |   [4]                 |     (3) (6)
            >>> b.insert(6)     |   [3, 6]              |     /   / \
            >>> b.insert(5)     |   [1, 5, 7]           |   (1) (5) (7)
            >>> b.insert(7)     |   [8]                 |             \
            >>> b.insert(8)     |                       |             (8)
        """
        def _find_parent(current, data):
            if current.data < data and current.right is not None:
                return _find_parent(current.right, data)
            elif current.data > data and current.left is not None:
                return _find_parent(current.left, data)
            elif current.data == data:
                raise ValueError(str(data) + " is already in the tree.")
            else:
                return current

        if self.root is None:
            self.root = BSTNode(data)
            return
        else:
            parent = _find_parent(self.root, data)

        if data < parent.data:
            parent.left = BSTNode(data)
            parent.left.prev = parent
        elif data > parent.data:
            parent.right = BSTNode(data)
            parent.right.prev = parent
        else:
            raise ValueError("SOMETHING WENT WRONG")

            
    
    # Problem 3: Implement BST.remove().
    def remove(self, data):
        """Remove the node containing 'data'. Consider several cases:
            - The tree is empty
            - The target is the root:
                - The root is a leaf node, hence the only node in the tree
                - The root has one child
                - The root has two children
            - The target is not the root:
                - The target is a leaf node
                - The target has one child
                - The target has two children
            If the tree is empty, or if there is no node containing 'data',
            raise a ValueError.
        """
                   

        if self.root is None:
            raise ValueError("The tree is empty.")
        
        node_to_remove = self.find(data)
        
        # no children
        if node_to_remove.left is None and node_to_remove.right is None:
            parent = node_to_remove.prev
            if node_to_remove.data > parent.data:
                parent.right = None
            else:
                parent.left = None

        # right child
        elif node_to_remove.left is None and node_to_remove.right is not None:
            parent = node_to_remove.prev
            if node_to_remove.data > parent.data:
                parent.right = node_to_remove.right
            else:
                parent.left = node_to_remove.right

        # left child
        elif node_to_remove.left is not None and node_to_remove.right is None:
            parent = node_to_remove.prev
            if node_to_remove.data > parent.data:
                parent.right = node_to_remove.left
            else:
                parent.left = node_to_remove.left

        # both children
        else:

            def _inorder_pred(current):
                if current.right is None:
                    data = current.data
                    self.remove(data)
                    return data
                else:
                    return _inorder_pred(current.right)

            temp = node_to_remove
            inorder_data = _inorder_pred(temp.left)
            node_to_remove.data = inorder_data
    

    def __str__(self):
        """String representation: a hierarchical view of the BST.
        Do not modify this function, but use it often to test this class.
        (this function uses a depth-first search; can you explain how?)
        
        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed out
                (2) (5)    [2, 5]       by depth levels. The edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        
        if self.root is None:                   # Print an empty tree
            return "[]"
        # If the tree is nonempty, create a list of lists.
        # Each inner list represents a depth level in the tree.
        str_tree = [list() for i in xrange(_height(self.root) + 1)]
        visited = set()                         # Track visited nodes
        
        def _visit(current, depth):
            """Add the data contained in 'current' to its proper depth level
            list and mark as visited. Continue recusively until all nodes have
            been visited.
            """
            str_tree[depth].append(current.data)
            visited.add(current)
            if current.left and current.left not in visited:
                _visit(current.left, depth+1)   # travel left recursively
            if current.right and current.right not in visited:
                _visit(current.right, depth+1)  # travel right recursively
        
        _visit(self.root,0)                     # Load the list of lists.
        out = ""                                # Build the final string.
        for level in str_tree:
            if level != list():                 # Ignore empty levels.
                out += str(level) + "\n"
            else:
                break
        return out

class AVL(BST):
    """AVL Binary Search Tree data structure class. Inherits from the BST
    class. Includes methods for rebalancing upon insertion. If your
    BST.insert() method works correctly, this class will work correctly.
    Do not modify.
    """
    def _checkBalance(self, n):
        if abs(_height(n.left) - _height(n.right)) >= 2:
            return True
        else:
            return False
    
    def _rotateLeftLeft(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.data > temp.data:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp
    
    def _rotateRightRight(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.data > temp.data:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n == self.root:
            self.root = temp
        return temp
    
    def _rotateLeftRight(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotateLeftLeft(n)
    
    def _rotateRightLeft(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotateRightRight(n)
    
    def _rebalance(self,n):
        """Rebalance the subtree starting at the node 'n'."""
        if self._checkBalance(n):
            if _height(n.left) > _height(n.right):
                # Left Left case
                if _height(n.left.left) > _height(n.left.right):
                    n = self._rotateLeftLeft(n)
                # Left Right case
                else:
                    n = self._rotateLeftRight(n)
            else:
                # Right Right case
                if _height(n.right.right) > _height(n.right.left):
                    n = self._rotateRightRight(n)
                # Right Left case
                else:
                    n = self._rotateRightLeft(n)
        return n
    
    def insert(self, data):
        """Insert a node containing 'data' into the tree, then rebalance."""
        # insert the data like usual
        BST.insert(self, data)
        # rebalance from the bottom up
        n = self.find(data)
        while n:
            n = self._rebalance(n)
            n = n.prev
    
    def remove(self, *args):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() has been disabled for this class.")

def _height(current):
    """Calculate the height of a given node by descending recursively until
    there are no further child nodes. Return the number of children in the
    longest chain down. Helper function for the AVL class and BST.__str__.
    Do not modify.
                                node | height
    Example:  (c)                  a | 0
              / \                  b | 1
            (b) (f)                c | 3
            /   / \                d | 1
          (a) (d) (g)              e | 0
                \                  f | 2
                (e)                g | 0
    """
    if current is None:     # Base case: the end of a branch.
        return -1           # Otherwise, descend down both branches.
    return 1 + max(_height(current.right), _height(current.left))

# =============================== END OF FILE =============================== #
