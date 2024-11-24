import pprint
from collections import defaultdict

# A simple class to represent graphs

class Graph(object):
    """ Graph data structure, undirected by default. """

    def __init__(self, connections=[], directed=True):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_all_edges(connections)

    def add_all_edges(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add_edge(node1, node2)

    def add_node(self, node):
        """ Add connection between node1 and node2 """
        if node not in self._graph:
            self._graph[node] = set()

    def add_edge(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def is_acyclic(self) -> bool:
        if self._directed:
            return self._is_acyclic_directed()
        else:
            return self._is_acyclic_undirected()
    
    def _is_acyclic_directed(self) -> bool:
        visited = set()
        recursion_stack = set()

        def dfs(node):
            if node in recursion_stack:  # Cycle detected
                return False
            if node in visited:
                return True  # Skip already verified nodes
            visited.add(node)
            recursion_stack.add(node)
        
            for neighbor in self.get_outgoing(node):
                if not dfs(neighbor):
                    return False
            recursion_stack.remove(node)
            return True

        # Check all nodes (in case of disconnected graph)
        return all(dfs(node) for node in self._graph if node not in visited)


    def _is_acyclic_undirected(self):
        visited = set()
    
        def dfs(node, parent):
            visited.add(node)
            for neighbor in self.get(node):
                if neighbor not in visited:
                    if not dfs(neighbor, node):
                        return False
                elif neighbor != parent:  # Cycle detected
                    return False
            return True
    
        # Check all connected components
        for node in self._graph:
            if node not in visited:
                if not dfs(node, None):
                    return False
        return True

    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
