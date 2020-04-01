from neural_net.data_handling import AdjData
from neural_net.edge_filters import matrix_size
from neural_net.network import Net
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Phys 490 Final Project: CQCNN')
    parser.add_argument('-d', dest='data_file', default='data/graphs_10.csv',
                        required=True, help='path to csv data file')
    parser.add_argument('-param', metavar='param.json', required=True, default='param/param_1.json',
                        help='path to json hyperparameter file')
    parser.add_argument('-v', type=int, default=2, metavar='N',
                        help='verbosity (default: 2)')
    args = parser.parse_args()

    # Hyperparameters from json file
    with open(args.param) as f:
        param = json.load(f)
    batch_size = param.get('batch_size', 10)
    num_epochs = param.get('epochs', 1)
    display_batch = param.get('display_batch', batch_size)
    display_epoch = param.get('display_epoch', 1)
    learn_rate = param.get('learning_rate', 0.001)

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
        CQCNN.backprop(data, loss_func, epoch+1, optimizer, display_epoch, display_batch, args.v)

    # test
    CQCNN.eval()
    CQCNN.test(data, batch_size, loss_func)
