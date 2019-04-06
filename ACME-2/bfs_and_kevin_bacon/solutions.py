
# Volume II Lab 7: Breadth-First Search

# Sean Wade


from collections import OrderedDict, deque
import networkx as nx
from matplotlib import pyplot as plt



class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    # Problem 1
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.
        
        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        total_str = str()
        sorted_dic = OrderedDict(sorted(self.dictionary.items(), key=lambda t: t[0]))
        for key in sorted_dic:
            total_str += "%s:" % (key)
            it = 0
            for item in self.dictionary[key]:
                if len(self.dictionary[key]) == (it+1):
                    total_str += " %s" % (item)
                else:
                    total_str += " %s;" % (item)
                it += 1
                
            total_str += "\n"
        return total_str
    
    
    # Problem 2
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        if not self.dictionary.has_key(start):
            raise ValueError("%s is not in the graph." % start)
            
        current = start
        marked = set()
        visited = list()
        visited_queue = deque()
        
        # visit the start node 'current'
        visited.append(current)
        marked.add(current)
        visited_queue.appendleft(current)
        
        while len(visited) < len(self.dictionary):
            current = visited_queue.popleft()
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visited.append(neighbor)
                    visited_queue.append(neighbor)
                    marked.add(neighbor)
                             
        return visited

    
    # Problem 3 (Optional)
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        if not self.dictionary.has_key(start):
            raise ValueError("%s is not in the graph." % start)
            
        current = start
        marked = set()
        visited = list()
        visited_queue = deque()
        
        # visit the start node 'current'
        
        marked.add(current)
        visited_queue.append(current)
        
        while len(visited) < len(self.dictionary):
            current = visited_queue.pop()
            visited.append(current)
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visited_queue.append(neighbor)
                    marked.add(neighbor)
                             
        return visited

    # Problem 4
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        if not self.dictionary.has_key(start):
            raise ValueError("%s is not in the graph." % start)
        elif not self.dictionary.has_key(target):
            raise ValueError("%s is not in the graph." % target)
        
        path_dict = {}
        current = start
        marked = set()
        visited = list()
        visited_queue = deque()
        path_list = []
        
        # visit the start node 'current'
        path_dict[current] = current
        visited.append(current)
        marked.add(current)
        visited_queue.appendleft(current)
        
        while len(visited) < len(self.dictionary):
            current = visited_queue.popleft()
            if current == target:
                break
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visited.append(neighbor)
                    visited_queue.append(neighbor)
                    marked.add(neighbor)
                    path_dict[neighbor] = current
        
        path_list.append(current)
        while current != start:
            current = path_dict[current]
            path_list.append(current)
            
            
        return path_list[::-1]
            
        
# Problem 5: Write the following function
def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph = nx.Graph()
    for key in dictionary:
        for value in dictionary[key]:
            nx_graph.add_edge(key, value)
    
    return nx_graph


# Helper function for problem 6
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)
    
    return graph



# Problems 6-8: Implement the following class
class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    # Problem 6
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        dictionary = parse(filename)
        self.nx_graph = convert_to_networkx(dictionary)
        self.actors = set()
        for x in dictionary:
            for y in dictionary[x]:
                self.actors.add(y)
            

    # Problem 6
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        return nx.shortest_path(self.nx_graph, start, target)

    # Problem 7
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        return (len(self.path_to_bacon(start, target)) - 1) / 2.0

    # Problem 7
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        total = 0
        total_people = 0
        not_connected = 0
        for person in self.actors:
            try:
                bn = self.bacon_number(person)
                total += bn
                total_people += 1
            except:
                not_connected += 1
                
        return total/float(total_people), not_connected
    
    # Problem 8
    def plot_bacon(self):
        total = []
        for person in self.actors:
            try:
                total.append(self.bacon_number(person))
            except:
                pass
                
        plt.hist(total, bins=6)
        plt.title("Bacon Number Distribution")
        plt.xlim([0,6])
        plt.xlabel("Bacon Number")
        plt.ylabel("Actors")
        plt.show()
                   

a = BaconSolver()
average, no = a.average_bacon()
print "The average bacon number is: %s" % average
print "The total who arn't: %s" % no
a.plot_bacon()





