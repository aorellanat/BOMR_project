class Node:
    def _init_(self, position, g_cost=float('inf'), h_cost=0):
        """
        Node class constructor that initializes:
        - position: current node position (x, y)
        - g_cost: cost from start to current node by default infinite
        - h_cost: estimated cost from current node to goal (heuristic) by default 0
        - parent: reference to parent node for path reconstruction
        - neighbors: list of (neighbor_node, cost) tuples
        """
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.parent = None
        self.neighbors = []
        
    def add_neighbor(self, neighbor, cost=1):
        """Add a neighbor node to the current node with a given cost"""
        self.neighbors.append((neighbor, cost))
        
    def get_neighbors(self):
        """Return the list of neighbors of the current node"""
        return self.neighbors

    def f_cost(self):
        """Return the sum of g_cost and h_cost"""
        return self.g_cost + self.h_cost
