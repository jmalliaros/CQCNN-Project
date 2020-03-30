from neural_net.data_handling import AdjData
from neural_net.edge_filters import matrix_size
from neural_net.network import Net
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neural_net.edge_filters import edge_to_edge

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Tutorial')
    parser.add_argument('-d', dest='data_file', metavar='data/graphs_10.csv',
                        help='data file name')
    args = parser.parse_args()

    num_epochs = 2
    learn_rate = 0.01
    batch_size = 1

    # instantiate the data object with a test size of 0.1017 (this can be changed)
    data = AdjData(csv_path=args.data_file, test_size=0.1017, batch_size=batch_size)

    # retrieve the dimensions of the adjacency matrix from the data file name
    dim = matrix_size(args.data_file)

    # pull data and convert to proper objects (in this case NxN matrices

    # X_train = torch.tensor(list(data.X_train.values)).view(-1,1,dim,dim).float()
    # Y_train = np.matrix(data.Y_train.tolist())
    # X_test = torch.tensor(list(data.X_test.values)).view(-1,1,dim,dim).float()
    # Y_test = np.matrix(data.Y_test.tolist())



    # instantiate model, optimizer and loss function
    CQCNN = Net(dim, batch_size)
    optimizer = optim.Adam(CQCNN.parameters(), lr=learn_rate)
    loss_func = nn.CrossEntropyLoss()

    # # reshape data into NxN matrices, not done using torch tensors because of custom conv layers
    # # TODO: do this in a parallelized way
    # X_train = []
    # for i in range(len(data.X_train.values) - 1):
    #     X_train.append(data.X_train.values[i].reshape(dim,dim))

    # convert training data into numpy array and apply edge-to-edge and and edge-to-vertex filters
    # X_train = np.array(X_train)
    # X_train = np.array([edge_to_edge(dim, m) for m in X_train])

    # convert to pytorch tensor for input into feed forward NN function, convert to floats
    # X_train = torch.from_numpy(X_train).float()
    CQCNN.train()
    for epoch in range(1, num_epochs):
        # for batch_id, (x, y) in enumerate(data.train_loader):
            # # reshape x to tensor with n rows and columns
            # output = CQCNN.forward(x.view(10, 10).float())
            # print(output)
            # print(output.shape)
            # print(y)
        CQCNN.backprop(data, loss_func, epoch, optimizer, 1, 1)


