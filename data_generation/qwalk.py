import pygame

from qtools import *
from graphs import *
from generate_graphs import random_adjacency_matrix

import matplotlib.pyplot as plt
import numpy as np

pygame_enabled = False

# qwalk_me(A)
# given a graph G defined by an adjacency matrix A,
# simulate a quantum walk (running indefinitely) on G
def qwalk_me(A, s, e):
    # switch graph from matrix to list
    G = matrixToList(A)

    c_index = s  # current index for this iteration
    count = 0  # count of transitions

    [num_rows, num_cols] = A.shape
    assert num_rows == num_cols

    # start time for quantum walk
    current_time = 0
    delta_time = 0.01

    # initialize vertex amplitudes and probabilities
    ampl = [(1.0 if i == 0 else 0.0) for i in range(num_rows)]
    probs = [(1.0 if i == 0 else 0.0) for i in range(num_rows)]

    while True:
        count += 1

        elements = np.arange(A.shape[0])

        # recalculate amplitudes at vertices
        U = qwalk(A, current_time)
        ampl = [U[i][s] for i in range(num_rows)]

        # recalculate probabilities at vertices (if measurement was taken now)
        probs = [(ampl[i] * ampl[i].conjugate()).real for i in range(num_rows)]


        sample = np.random.choice(elements, p=probs)  # sample a target using probs
        #print(count,probs, sample)
        c_index = sample  # go to target
        if c_index == e:  # if target is our ending point
            #print('final count', count)
            return count, probs  # stop walking
        elif count > 250:
            return count, probs

        # update time
        current_time += delta_time

