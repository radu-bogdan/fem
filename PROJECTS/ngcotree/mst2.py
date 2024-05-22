# Python program for Kruskal's algorithm to find 
# Minimum Spanning Trees of given connected or disconnected, 
# undirected and weighted graph 

# Class to represent a graph 
class Graph: 
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [] 
  
    # Function to add an edge to graph 
    def addEdge(self, u, v, w): 
        self.graph.append([u, v, w]) 
  
    # A utility function to find set of an element i 
    # (truly uses path compression technique) 
    def find(self, parent, i): 
        if parent[i] != i: 
            parent[i] = self.find(parent, parent[i]) 
        return parent[i] 
  
    # A function that does union of two sets of x and y 
    # (uses union by rank) 
    def union(self, parent, rank, x, y): 
        if rank[x] < rank[y]: 
            parent[x] = y 
        elif rank[x] > rank[y]: 
            parent[y] = x 
        else: 
            parent[y] = x 
            rank[x] += 1
  
    # The main function to construct MSTs 
    # using Kruskal's algorithm 
    def KruskalMST(self): 
        result = []  # This will store the resultant MST(s)
        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]

        # Sort all the edges in non-decreasing order of their weight
        self.graph = sorted(self.graph, key=lambda item: item[2]) 
  
        parent = []
        rank = []
  
        # Create V subsets with single elements
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
  
        # Number of edges to be taken is less than V-1
        while e < self.V - 1 and i < len(self.graph): 
            u, v, w = self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent, v) 
  
            # If including this edge doesn't cause cycle
            if x != y: 
                e = e + 1
                result.append([u, v, w]) 
                self.union(parent, rank, x, y) 
  
        # Check for any remaining components that might be disconnected
        components = {}
        for node in range(self.V):
            root = self.find(parent, node)
            if root in components:
                components[root].append(node)
            else:
                components[root] = [node]
  
        # Print results for each component
        totalCost = 0
        print("Edges in the constructed MST(s):")
        for root, nodes in components.items():
            component_cost = 0
            print(f"Component with root {root}:")
            for u, v, weight in result:
                if self.find(parent, u) == root:
                    component_cost += weight
                    print(f"{u} -- {v} == {weight}")
            totalCost += component_cost
            print(f"Component cost: {component_cost}")
        
        print(f"Total Minimum Spanning Tree cost: {totalCost}")
        self.MST = result

# Driver code 
if __name__ == '__main__': 
    g = Graph(6) 
    g.addEdge(0, 1, 10) 
    g.addEdge(0, 2, 6) 
    g.addEdge(0, 3, 5) 
    g.addEdge(1, 3, 15) 
    g.addEdge(2, 3, 4) 
    g.addEdge(4, 5, 1) # Adding an edge for a disconnected component

    # Function call 
    g.KruskalMST() 