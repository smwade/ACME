# name this file solutions.py

"""Volume II Lab 6: Nearest Neighbor Search
Sean Wade
MATH 321
10/10/15
"""

from Trees import BST
from Trees import BSTNode
from math import sqrt
import numpy as np
from scipy.spatial import distance, KDTree
from sklearn import neighbors


# Problem 1: Implement this function.
def euclidean_metric(x, y):
    """Return the euclidean distance between the vectors 'x' and 'y'.

    Raises:
        ValueError: if the two vectors 'x' and 'y' are of different lengths.
    
    Example:
        >>> print(euclidean_metric([1,2],[2,2]))
        1.0
        >>> print(euclidean_metric([1,2,1],[2,2]))
        ValueError: Incompatible dimensions.
    """
    if len(x) != len(y):
        raise ValueError("Incompatible dimensions.")
    return distance.euclidean(x, y)

# print euclidean_metric(np.array([2,3]), np.array([1,1]))

        

# Problem 2: Implement this function.
def exhaustive_search(data_set, target):
    """Solve the nearest neighbor search problem exhaustively.
    Check the distances between 'target' and each point in 'data_set'.
    Use the Euclidean metric to calculate distances.
    
    Inputs:
        data_set (mxk ndarray): An array of m k-dimensional points.
        target (1xk ndarray): A k-dimensional point to compare to 'dataset'.
        
    Returns:
        the member of 'data_set' that is nearest to 'target' (1xk ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """

    if not isinstance(data_set, np.ndarray) or not isinstance(target, np.ndarray):
        raise TypeError("Should be np.ndarray")



    best_path = -1
    best_distance = 999999999999999999.9
    if len(data_set.shape) > 1:
        rows = data_set.shape[0]
        cols = data_set.shape[1]
    else:
        rows, cols = data_set.shape[0]

    data_set.astype(float)
    target.astype(float)
    if cols != target.shape[0]:
        raise ValueError("Incompatible dimensions.")
    for i in xrange(0, rows):
        dst = euclidean_metric(data_set[i], target)
        if dst < best_distance:
            best_distance = dst
            best_path = i
    return data_set[best_path], best_distance

# a, b =  exhaustive_search(np.array([[1,2], [3,89], [4,4]]), np.array([9,8]))


# Problem 3: Finish implementing this class by modifying __init__()
#   and adding the __sub__, __eq__, __lt__, and __gt__ magic methods.
class KDTNode(BSTNode):
    """Node class for K-D Trees. Inherits from BSTNode.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        parent (KDTNode): a reference to this node's parent node.
        data (ndarray): a coordinate in k-dimensional space.
        axis (int): the 'dimension' of the node to make comparisons on.
    """

    def __init__(self, data):
        """Construct a K-D Tree node containing 'data'. The left, right,
        and prev attributes are set in the constructor of BSTNode.

        Raises:
            TypeError: if 'data' is not a a numpy array (of type np.ndarray).
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("should be np.ndarray")
        BSTNode.__init__(self, data)
        self.axis  = 0

    def __sub__(self, other):
        return euclidean_metric(self.data, other.data)

    def __eq__(self, other):
        if np.allclose(self.data, other.data):
            return True
        else:
            return False

    def __lt__(self, other):
        return self.data[other.axis] < other.data[other.axis]

    def __gt__(self, other):
        return self.data[other.axis] > other.data[other.axis]

A = KDTNode(np.array([1,2]))
B = KDTNode(np.array([3,1]))
#print A < B
#print A > A
#B.axis = 1
#print A < B
#print A > B




# Problem 4: Finish implementing this class by overriding
#   the insert() and remove() methods.
class KDT(BST):
    """A k-dimensional binary search tree object.
    Used to solve the nearest neighbor problem efficiently.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other
            nodes in the tree, the root houses data as a numpy array.
        k (int): the dimension of the tree (the 'k' of the k-d tree).
    """

    def __init__(self, data_set):
        """Set the k attribute and fill the tree with the points in 'data_set'.

        Raises:
            TypeError: if 'data_set' is not a numpy array (of type np.ndarray)
        """

        # Validate the input type.
        if not isinstance(data_set, np.ndarray):
            raise TypeError("data_set must be a numpy array.")

        # Set the root and dimension attributes.
        BST.__init__(self)
        self.k = data_set.shape[1]

        # Load the data into the tree one point at a time.
        for point in data_set:
            self.insert(point)
    
    def find(self, data):
        """Return the node containing 'data'.

        Raises:
            ValueError: if there is node containing 'data' in the tree,
                or the tree is empty.
        """

        # First check that the tree is not empty.
        if self.root is None:
            raise ValueError(str(data) + " is not in the tree.")
        
        # Define a recursive function to traverse the tree.
        def _step(current, target):
            """Recursively approach the target node."""
            
            if current is None:             # Base case: target not found.
                return current
            if current == target:            # Base case: target found!
                return current
            if target < current:            # Recursively search to the left.
                return _step(current.left, target)
            else:                           # Recursively search to the right.
                return _step(current.right, target)
            
        # Create a new node to use the KDTNode comparison operators.
        n = KDTNode(data)

        # Call the recursive function, starting at the root.
        found = _step(self.root, n)
        if found is None:                  # Report the data was not found.
            raise ValueError(str(data) + " is not in the tree.")
        return found                       # Otherwise, return the target node.
    
    def insert(self, data):
        """Inserts new node containing 'data' """
        new_node = KDTNode(data)
        
        if self.root is None:
            self.root = new_node
            self.axis = 0
            return

        current = self.root
        temp = None
        while current is not None:
            temp = current
            if new_node < current:
                # if current.left is None:
                    #break
                current = current.left
            else:
                #if current.right is None:
                    #break
                current = current.right

        if new_node < temp:
            temp.left = new_node
        else:
            temp.right = new_node
        new_node.axis = (temp.axis + 1) % self.k
        new_node.prev = temp

    def remove(*args, **kwargs):
        raise NotImplementedError("remove is disabled for this.")



# Problem 5: Implement this function.
def nearest_neighbor(data_set, target):
    """Use your KDTree class to solve the nearest neighbor problem.

    Inputs:
        tree (KDT): A KDT object loaded with a data set of m
            k-dimensional points (as numpy arrays).
        target (1xk ndarray): A k-dimensional point to compare to the
            data housed in 'tree'.

    Returns:
        The point in the tree that is nearest to 'target' (1xk ndarray).
        The distance from the nearest neighbor to 'target' (float).
    """
    # should take a data_set instead of tree

    def kdts(current, target, neighbor, distance):
        index = current.axis
        if euclidean_metric(current.data, target.data) < distance:
            neighbor = current
            distance = euclidean_metric(current.data, target.data)

        if target.data[index] < current.data[index]:
            if current.left is not None:
                neighbor, distance = kdts(current.left, target, neighbor, distance)
            if target.data[index] + distance >= current.data[index]:
                if current.right is not None:
                    neighbor, distance = kdts(current.right, target, neighbor, distance)
        else:
            if current.right is not None:
                neighbor, distance = kdts(current.right, target, neighbor, distance)
            if target.data[index] - distance <= current.data[index]:
                if current.left is not None:
                    neighbor, distance = kdts(current.left, target, neighbor, distance)
        return neighbor, distance
    

    tree = KDT(data_set)
    current = tree.root
    target = KDTNode(target)
    distance = euclidean_metric(current.data, target.data)
    a, b = kdts(current, target, current, distance)
    return a.data, b

if __name__ == "__main__":
    data_set = np.random.random((100, 20))
    target = np.random.random(20)
    tree = KDTree(data_set)

    dis, n = tree.query(target)
    a, b = nearest_neighbor(data_set, target)
    print "+++++++++++++++++++++++++++++++"
    print tree.data[n]
    print a






# Problem 6: Implement this function.
def postal_problem():
    """Use the neighbors module in sklearn to classify the Postal data set
    provided in 'PostalData.npz'. Classify the testpoints with 'n_neighbors'
    as 1, 4, or 10, and with 'weights' as 'uniform' or 'distance'. For each
    trial print a report indicating how the classifier performs in terms of
    percentage of misclassifications.

    Your function should print a report similar to the following:
    n_neighbors = 1, weights = 'distance':  0.903
    n_neighbors = 1, weights =  'uniform':  0.903       (...and so on.)
    """

    def test_right(x, w):
        labels, points, testlabels, testpoints = np.load('PostalData.npz').items()
        nbrs = neighbors.KNeighborsClassifier(n_neighbors=x, weights=w, p=2)
        nbrs.fit(points[1], labels[1])
        prediction = nbrs.predict(testpoints[1])
        result = np.average(prediction/testlabels[1])
        print "n_neighbors = %s, weights = %s:   %s" % (x, w, result)
    
    test_right(1,'distance')
    test_right(1,'uniform')
    test_right(4,'distance')
    test_right(4,'uniform')
    test_right(10,'distance')
    test_right(10,'uniform')
   
# =============================== END OF FILE =============================== #
