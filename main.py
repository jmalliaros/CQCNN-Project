import random
import numpy as np
import csv
import os
from generate_graphs import *
from classical_walk import random_walk

n = 10
size = 1000


def write_csv(graph_data, n):
    folder_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder_path, 'data/graphs_{}.csv'.format(n))
    with open(file_path, "w", newline='') as f:
        graph_writer = csv.writer(f, delimiter = ' ',)

        for graph in graph_data:
            print(graph)
            classical_count = random_walk(*graph)
            quantum_count = qwalk_count_placeholder(*graph)
            # row = np.append(graph[0].reshape(-1), np.array([graph[1], graph[2], classical_count, quantum_count]))
            row = graph[0].reshape(-1).tolist() + [graph[1], graph[2], classical_count, quantum_count]
            print(row)
            graph_writer.writerow(row)


if __name__ == '__main__':
    graph_dataset = generate_graph_data(n, size)
    write_csv(graph_dataset, n)
