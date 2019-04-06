from LinkedLists import Deque
from LinkedLists import Queue
import unittest

def fun(x):
    return x + 1

class DequeTestCases(unittest.TestCase):
    def test_add(self):
        a = Deque()
        a.append(5)
        self.assertEqual(a.pop(), 5)

class QueueTestCases(unittest.TestCase):
    def testadd(self):
        a = Queue()
        a.push(5)
        self.assertEqual(a.pop(), 5)
        self.assertRaises(ValueError)
        
if __name__ == '__main__':
    unittest.main()
