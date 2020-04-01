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


def sum(M,n,i,j):
    """
    Summation in edge-to-edge filtering helper function
    :param M: matrix
    :param n: size n
    :param i: row index
    :param j: column index
    :return: matrix
    """
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][j]
    return x


def sum_ev(M,n,i):
    """
    Summation for edge-to-vertex filtering helper function
    :param M: matrix
    :param n: size n
    :param i: index
    :return: matrix
    """
    x = 0
    for k in range(n):
        x += M[i][k] + M[k][i]
    return x


#
def edge_to_vertex(F, n):
    """
    Edge to vertex filter, initiate empty array first
    :param F: matrix representing graph
    :param n: number of vertices in graph
    :return: size nx1 vector representing the sum of each vertex node
    """
    Fev = torch.zeros([1, n], dtype=torch.float)
    for i in range(n):
        Fev[0][i] = sum_ev(F,n,i) - 2*F[i][i]
    return Fev


def edge_to_edge(dim, adj_matrix):
    """
    Edge to Edge filter, initiate empty matrix
    :param dim: number of node vertices in graph
    :param adj_matrix: adjacency matrix representing graph
    :return: adjacency matrix
    """
    Fee = torch.zeros([dim, dim], dtype=torch.float)
    for i in range(dim):
        for j in range(dim):
            Fee[i][j] = (sum(adj_matrix,dim,i,j) - 2*adj_matrix[i][j])*adj_matrix[i][j]
    return Fee

