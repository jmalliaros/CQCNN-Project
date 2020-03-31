import random
import numpy as np


def random_adjacency_matrix(n):
    """
    creates an nxn symmetric adjacency matrix with 0s along the diagonal
    used to represent an undirected graph with n nodes
    :param n: dimension
    :return: nxn numpy array
    """
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]

    return np.array(matrix)


def rand_start_end_points(matrix):
    """
    generates start and end point of walk randomly
    :param matrix: adjacency matrix
    :return: tuple of start and end nodes
    """
    n = len(matrix)
    s, e = random.sample(range(0, n), 2)
    # can't have start and end to be the same
    while e == s:
        e = random.randrange(0, n)
    return s, e


def far_start_end_points(matrix):
    """
    generates start and end points on different ends of matrix
    :param matrix: adjacency matrix
    :return: tuple of start and end nodes
    """
    n = len(matrix)
    s = random.randrange(0, n//2 - 1)
    e = random.randrange(n//2 + 1, n)
    return s, e

def generate_graph_data(n, size):
    """
    generate adjacency matrix with start and end points
    :param n: number of nodes
    :param size: dataset size
    :return: list of tuples (matrix: nxn size np.array, start: int, end: int)
    """
    dataset = []
    for i in range(size):
        matrix = random_adjacency_matrix(n)
        # checks if matrix contains n vertices, if list contains False, zeroes row exists,
        # will need to recreate matrix with n vertices
        while False in matrix.any(axis=1):
            matrix = random_adjacency_matrix(n)
        start, end = far_start_end_points(matrix)
        dataset.append((matrix, start, end))
    return dataset
