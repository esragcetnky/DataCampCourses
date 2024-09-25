from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import nxviz as nv


T = nx.Graph()

T.add_nodes_from((1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49))


T.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), 
                  (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), 
                  (1, 23), (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 32), 
                  (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 42), 
                  (1, 43), (1, 44), (1, 45), (1, 46), (1, 47), (1, 48), (1, 49), (5, 19), (5, 28), (5, 36), 
                  (7, 28), (8, 19), (8, 28), (11, 19), (11, 28), (13, 19), (14, 28), (15, 19), (15, 28), (16, 18), 
                  (16, 35), (16, 36), (16, 48), (17, 19), (17, 28), (18, 24), (18, 35), (18, 36), (19, 20), (19, 21), 
                  (19, 24), (19, 30), (19, 31), (19, 35), (19, 36), (19, 37), (19, 48), (20, 28), (21, 28), (24, 28), 
                  (24, 36), (24, 37), (24, 39), (24, 43), (25, 28), (27, 28), (28, 29), (28, 30), (28, 31), (28, 35), 
                  (28, 36), (28, 37), (28, 44), (28, 48), (28, 49), (29, 43), (33, 39), (35, 36), (35, 37), (35, 39), 
                  (35, 43), (36, 37), (36, 39), (36, 43), (37, 43), (38, 39), (39, 40), (39, 41), (39, 45), (41, 45), (43, 47), (43, 48)])

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(list(G.neighbors(n)), 2):

        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1,n2):

            in_open_triangle = True

            break

    return in_open_triangle

# Compute the number of open triangles in T
num_open_triangles = 0

# Iterate over all the nodes in T
for n in T.nodes():

    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):

        # Increment num_open_triangles
        num_open_triangles += 1

print(num_open_triangles)