import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from neural_net.edge_filters import edge_to_edge

class Net(nn.Module):
    """
    Class to define fully connected layer NN
    """
    def __init__(self, dim):
        super(Net, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x_1 = self.fc1(x)
        x_2 = func.relu(x_1)
        return self.fc2(x_2)

    def test(self, X_test, Y_test, loss):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test)
            targets = torch.from_numpy(Y_test)
            output = self.forward(inputs)
            cross_val = loss(output, targets.reshape(-1))
        return cross_val.item(), output, targets

