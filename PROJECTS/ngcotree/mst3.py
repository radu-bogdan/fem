# Python program for Kruskal's algorithm to find 
# Minimum Spanning Trees of given connected or disconnected, 
# undirected graph, with edges provided in a 2xN array

# Class to represent a graph 
class Graph: 
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [] 
        self.mst_list = [] # Attribute to store the MSTs for each component
  
    # Function to add edges to the graph from a 2xN array
    def addEdges(self, edge_array): 
        # Each column in edge_array represents an edge
        for u, v in zip(edge_array[0], edge_array[1]):
            self.graph.append((u, v)) 
  
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

        # Sort all the edges by their index
        indexed_graph = [(u, v, idx) for idx, (u, v) in enumerate(self.graph)]
        indexed_graph.sort(key=lambda item: item[2]) 
  
        parent = []
        rank = []
  
        # Create V subsets with single elements
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
  
        # Number of edges to be taken is less than V-1
        while i < len(indexed_graph): 
            u, v, idx = indexed_graph[i] 
            i += 1
            x = self.find(parent, u) 
            y = self.find(parent, v) 
  
            # If including this edge doesn't cause cycle
            if x != y: 
                result.append((u, v, idx)) 
                self.union(parent, rank, x, y) 
  
        # Collect results for each component
        self.mst_list = []  # Reset mst_list to store current MSTs
        # print("Edges in the constructed MST(s):")
        for u, v, idx in result:
            # print(f"{u} -- {v} (index {idx})")
            self.mst_list.append(idx)
        
        # print("MST Edge Indices:", self.mst_list)
        return self.mst_list

# Driver code 
if __name__ == '__main__': 
    g = Graph(6)
    edge_array = [[0, 0, 0, 1, 2, 4], 
                  [1, 2, 3, 3, 3, 5]]  # Example 2xN edge array
    g.addEdges(edge_array)

    # Function call 
    mst_indices = g.KruskalMST()
    
    # Accessing the MST edge indices
    print("\nList of MST edge indices:")
    print(mst_indices)

# This code is contributed by Neelam Yadav 
# Improved by James Graça-Jones 
# Modified by [Your Name]
