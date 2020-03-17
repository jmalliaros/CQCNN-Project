import random
import numpy as np
import matplotlib.pyplot as plt
import time


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


def normalize_row(matrix: np.array):
    """
    normalizes each row by dividing each elements of row i by the sum of row i
    :param matrix:
    :return: normalized matrix
    """
    matrix = matrix.astype(float)
    for i in range(len(matrix)):
        row_sum = sum(matrix[i])
        matrix[i] = matrix[i] / row_sum if row_sum != 0 else matrix[i] / 1
    return matrix


def random_walk(a, i):
    """
    computes the time it takes for a random walk to return to the starting node
    :param a: adjacency matrix
    :param i: starting index (starting node)
    :return: time (s) to return to starting node
    """
    # a -> adj
    # i -> starting row
    walk = []  # holds transitions
    elements = np.arange(a.shape[0])  # for our graph [0,1,2,3]
    c_index = i  # current index for this iteration
    count = 0  # count of transitions
    start = time.perf_counter()

    while True:
        count += 1
        probs = a[c_index]  # probability of transitions
        # sample from probs
        sample = np.random.choice(elements, p=probs)  # sample a target using probs
        c_index = sample  # go to target
        if sample == i:  # if target is our initial point
            walk.append(count)  # stop walking
            end = time.perf_counter()
            break
    return "%.2gs" % (end - start)


def random_walk_iter(a, i, iters):
    # a -> adj
    # i -> starting row
    walk = []  # holds transitions
    elements = np.arange(a.shape[0])  # for our graph [0,1,2,3]
    for k in range(iters):
        c_index = i  # current index for this iteration
        count = 0  # count of transitions
        while True:
            count += 1
            probs = a[c_index]  # probability of transitions
            # sample from probs
            sample = np.random.choice(elements, p=probs)  # sample a target using probs
            c_index = sample  # go to target
            if sample == i:  # if target is our initial point
                walk.append(count)  # stop walking
                break

    return walk






if __name__ == '__main__':
    adj = random_adjacency_matrix(10)
    print(adj)
    adj = normalize_row(adj)
    print(adj)
    # adj = np.array([
    #     [0, 1, 1, 0],
    #     [1, 0, 1, 0],
    #     [1, 1, 0, 1],
    #     [0, 0, 1, 0]
    # ], dtype=np.float32)
    # adj = normalize_row(adj)
    # counts = np.unique(random_walk_iter(adj, 0, 1000), return_counts=True)  # perform 1k walks
    # plt.bar(counts[0], counts[1])
    # plt.xticks(np.arange(counts[0][-1]), rotation=-60)
    # plt.show()
    print(random_walk(adj, 0))
