# Python program for Kruskal's algorithm to find 
# Minimum Spanning Trees of given connected or disconnected, 
# undirected graph, with edges provided as an Nx3 list

import random

# Class to represent a graph 
class Graph: 
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [] 
        self.mst_list = [] 
  
    def addEdges(self, edge_list): 
        for edge in edge_list:
            self.graph.append(tuple(edge))
  
    def find(self, parent, i): 
        if parent[i] != i: 
            parent[i] = self.find(parent, parent[i]) 
        return parent[i] 
  
    def union(self, parent, rank, x, y): 
        if rank[x] < rank[y]: 
            parent[x] = y 
        elif rank[x] > rank[y]: 
            parent[y] = x 
        else: 
            parent[y] = x 
            rank[x] += 1
    

    def KruskalMST(self, randomize=False): 
        result = []
        if randomize:
            random.shuffle(self.graph)  # Shuffle the edges if randomization is requested
        else:
            self.graph.sort(key=lambda item: item[2])  # Sort by the provided index otherwise
        
        parent = []
        rank = []
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
        i = 0
        while i < len(self.graph): 
            u, v, idx = self.graph[i] 
            i += 1
            x = self.find(parent, u) 
            y = self.find(parent, v) 
            if x != y: 
                result.append(idx) 
                self.union(parent, rank, x, y) 
        self.mst_list = result
        return self.mst_list

    # def KruskalMST(self): 
    #     result = []
    #     self.graph.sort(key=lambda item: item[2]) 
    #     parent = []
    #     rank = []
    #     for node in range(self.V): 
    #         parent.append(node) 
    #         rank.append(0) 
    #     i = 0
    #     while i < len(self.graph): 
    #         u, v, idx = self.graph[i] 
    #         i += 1
    #         x = self.find(parent, u) 
    #         y = self.find(parent, v) 
    #         if x != y: 
    #             result.append(idx) 
    #             self.union(parent, rank, x, y) 
    #     self.mst_list = result
    #     return self.mst_list

# Driver code 
if __name__ == '__main__': 
    g = Graph(4)
    edge_list = [
        [0, 1, 10], # [vertex u, vertex v, index]
        [0, 2, 20],
        [0, 3, 30],
        [1, 3, 40],
        [2, 3, 50]
    ]
    g.addEdges(edge_list)

    # Function call 
    mst_indices = g.KruskalMST()
    
    # Accessing the MST edge indices
    print("\nList of MST edge indices:")
    print(mst_indices)

# This code is contributed by Neelam Yadav 
# Improved by James Graça-Jones 
# Modified by [Your Name]
