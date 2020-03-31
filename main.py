from neural_net.data_handling import AdjData
from neural_net.edge_filters import matrix_size
from neural_net.network import Net
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('-d', dest='data_file', metavar='data/graphs_10.csv',
                        help='data file name')
    args = parser.parse_args()

    # TODO: refactor hyper parameters to json file
    num_epochs = 10
    learn_rate = 0.001
    batch_size = 32

    # instantiate the data object with a test size of 0.1017 (this can be changed)
    data = AdjData(csv_path=args.data_file, test_size=0.1017, batch_size=batch_size)

    # retrieve the dimensions of the adjacency matrix from the data file name
    dim = matrix_size(args.data_file)

    # instantiate model, optimizer and loss function
    CQCNN = Net(dim, batch_size)
    optimizer = optim.Adam(CQCNN.parameters(), lr=learn_rate)
    loss_func = nn.CrossEntropyLoss()

    CQCNN.train()
    for epoch in range(num_epochs):
        CQCNN.backprop(data, loss_func, epoch+1, optimizer, 1, 1)

    # test
    CQCNN.eval()
    CQCNN.test(data, batch_size, loss_func)
