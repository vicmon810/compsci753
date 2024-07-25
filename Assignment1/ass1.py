import numpy as np


# Read edge list from file
def read_edge_list(file_path):
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            source, target = map(int, line.strip().split())
            edges.append((source, target))
    return edges

file_path = 'web-Google-final.txt' 
print( read_edge_list(file_path))