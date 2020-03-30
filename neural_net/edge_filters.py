import numpy as np 
import re
import torch
import torch.nn as nn

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
def _edge_to_vertex(F,n):
    Fev = np.zeros([1,n])[0]
    for i in range(n):
        Fev[i] = sum_ev(F,n,i) - 2*F[i][i]
    return Fev

# x is the index of the graph in the data file
def _edge_to_edge(dim, adj_matrix):
    Fee = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(dim):
            Fee[i][j] = (sum(adj_matrix,dim,i,j) - 2*adj_matrix[i][j])*adj_matrix[i][j]
            Fev = edge_to_vertex(Fee,dim)
    return Fev



# Edge to edge filter, initiate empty array first
def edge_to_vertex(F,n):
    Fev = torch.zeros([1, n], dtype=torch.float)
    for i in range(n):
        Fev[0][i] = sum_ev(F,n,i) - 2*F[i][i]
    return Fev

# x is the index of the graph in the data file
def edge_to_edge(dim, adj_matrix):

    Fee = torch.zeros([dim, dim], dtype=torch.float)
    Fev = None
    for i in range(dim):
        for j in range(dim):
            Fee[i][j] = (sum(adj_matrix,dim,i,j) - 2*adj_matrix[i][j])*adj_matrix[i][j]
            Fev = edge_to_vertex(Fee,dim)
    # return torch.from_numpy(Fee)
    return Fee

if __name__ == '__main__':
    matrix = torch.from_numpy(np.asarray([[0,0,0,1], [0,0,1,1], [0,1,0,0], [1,1,0,0]])).float()
    print(edge_to_edge(4, matrix))
    matrix = np.tril(matrix)
    print(matrix)
    print(edge_to_vertex(matrix, 4))

    A = torch.from_numpy(np.asarray([[ 1.8939e+00, -7.7798e-01, -2.1888e-02,  6.4004e-02,  6.4004e-02,
            6.4004e-02,  6.4004e-02,  6.4004e-02],
          [-1.1964e+00, -4.5975e-02, -2.7878e-03, -2.1888e-02,  6.4004e-02,
            6.4004e-02,  6.4004e-02,  6.4004e-02],
          [-2.9751e-01, -2.3122e+00,  1.0821e+00, -9.9352e-01, -2.1888e-02,
            6.4004e-02,  6.4004e-02,  6.4004e-02],
          [-1.2436e+00,  9.7963e-01, -3.3408e+00,  2.0878e+00, -1.1013e+00,
           -2.1888e-02,  6.4004e-02,  6.4004e-02],
          [-2.7524e+00, -1.8605e+00, -3.4090e-01, -2.8969e+00,  9.9310e-01,
           -9.9352e-01, -2.1888e-02,  6.4004e-02],
          [-2.3627e+00, -1.3704e+00, -2.6963e+00, -3.3622e-02, -3.3915e+00,
            2.0878e+00, -1.1013e+00, -2.1888e-02],
          [-3.8317e-01, -4.0031e-01, -8.1301e-01, -3.1045e+00,  1.4940e+00,
           -4.2292e+00,  2.1904e+00, -1.1013e+00],
          [-1.0966e+00, -2.7588e-01, -2.4040e+00, -1.4336e+00, -3.3472e+00,
           -1.1024e+00, -3.0543e+00,  9.9310e-01]]))
    print(A.shape)
    e2v = edge_to_vertex(A, 8)
    print(e2v)
    print(e2v.shape)