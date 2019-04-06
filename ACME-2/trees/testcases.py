import unittest
from Trees import BST
import solutions as s
from LinkedList import LinkedList

class solution_tests(unittest.TestCase):

    def test_1(self):
        A = LinkedList()
        A.add(1)
        A.add(2)
        A.add(6)
        A.add(3)
        res = s.recursive_search(A, 2)
        self.assertIsNot(res,None)

    def test_2(self):
        A = BST()
        A.insert(4)
        A.insert(3)
        A.insert(6)
        A.insert(5)
        A.insert(7)
        A.insert(8)
        A.insert(1)

    def test_3(self):
        A = BST()
        A.insert(3)
        A.insert(2)
        A.insert(1)
        A.insert(8)
        A.insert(9)
        A.insert(5)
        A.insert(6)
        print A
        A.remove(8)
        print A




if __name__ == '__main__':
    unittest.main()
