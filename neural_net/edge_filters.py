import numpy as np 
import re


# Getting the matrix dimensions from the file name
regex = re.compile(r'\d+')
def matrix_size(csv_path):
    n = [int(x) for x in regex.findall(csv_path)][0]
    return n

# Length of data (i.e., 1000)
def length_data(data_file):
    data = np.loadtxt(data_file)
    return len(data)


# Getting adjacency matrix of one graph
# def data_to_graph(train, i, dim):
#     A = np.reshape(train[i], [dim, dim])
#     return A


# Summation in edge-to-edge filtering
def sum(M,n,i,j):
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][j]
    return x


# Summation for edge-to-vertex filtering
def sum_ev(M,n,i):
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][i]
    return x


# Edge to edge filter, initiate empty array first
def edge_to_vertex(F,n):
    Fev = np.zeros([1,n])[0]
    for i in range(n):
        Fev[i] = sum_ev(F,n,i) - 2*F[i][i]
    return Fev


# x is the index of the graph in the data file
def edge_to_edge(dim, adj_matrix):
    Fee = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            Fee[i][j] = (sum(adj_matrix,dim,i,j) - 2*adj_matrix[i][j])*adj_matrix[i][j]
            Fev = edge_to_vertex(Fee,dim)
    return Fev
