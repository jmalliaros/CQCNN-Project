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
    # can't have starting point on a row without any vertices (i.e. row is all 0s)
    while not np.any(matrix[s]):
        s = random.randrange(0, n)
    # can't have start and end to be the same
    while not np.any(matrix[e]) or e == s:
        e = random.randrange(0, n)
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
        start, end = rand_start_end_points(matrix)
        dataset.append((matrix, start, end))
    return dataset


# placeholder function for quantum walk
def qwalk_count_placeholder(matrix, start, end):
    return 9

# if __name__ == '__main__':
#     data = generate_graph_data(5, 10)
#     print(data)