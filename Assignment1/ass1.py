import networkx as nx
import numpy as np
import time

def get_google_matrix(G, d=0.15):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G).T
    # for sink nodes
    is_sink = np.sum(A, axis=0) == 0
    B = (np.ones_like(A) - np.identity(n)) / (n-1)
    A[:, is_sink] += B[:, is_sink]
    
    D_inv = np.diag(1 / np.sum(A, axis=0))
    M = np.dot(A, D_inv) 
    
    # for disconnected components
    M = (1 - d) * M + d * np.ones((n, n)) / n
    return M

def l1(x):
    return np.sum(np.abs(x))

def pagerank_edc(G, d=0.15):
    M = get_google_matrix(G, d=d)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    idx = eigenvalues.argsort()[-1]
    largest = np.array(eigenvectors[:,idx]).flatten().real
    return largest / l1(largest)

def pagerank_power(G, d=0.15, max_iter=100, eps=1e-9):
    M = get_google_matrix(G, d=d)
    n = G.number_of_nodes()
    V = np.ones(n) / n
    start_time = time.time()
    for i in range(max_iter):
        V_last = V.copy()
        V = np.dot(M, V)
        if l1(V - V_last) / n < eps:
            running_time = time.time() - start_time
            return V, running_time, i + 1
    running_time = time.time() - start_time
    return V, running_time, max_iter

def gen_webgraph(n, m):
    G = nx.DiGraph(nx.barabasi_albert_graph(n, m))
    rands = np.random.choice(n, n // 2, replace=False)
    G.remove_edges_from(np.array(G.edges)[rands])
    return G

# Generate a web graph
n = 1000  # number of nodes
m = 5     # number of edges to attach from a new node to existing nodes
G = gen_webgraph(n, m)

# Run the power iteration algorithm
rank_vector, running_time, num_iterations = pagerank_power(G)

# Get the top-10 ranked nodes
top_10_indices = np.argsort(-rank_vector)[:10]
top_10_scores = rank_vector[top_10_indices]

# Report the results
print("Running time:", running_time)
print("Number of iterations:", num_iterations)
print("Top-10 ranked nodes (ID, Score):")
for idx, score in zip(top_10_indices, top_10_scores):
    print(f"Node ID: {idx}, Score: {score}")
