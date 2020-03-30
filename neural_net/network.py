import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from neural_net.edge_filters import edge_to_edge, edge_to_vertex
from torch.autograd import Variable


class convVert(nn.Module):
    # edge to vertex filter
    def __init__(self):
        super(convVert, self).__init__()

    def forward(self, x):
        # shape of x is [batch_size, 1, dim, dim]
        # dim and length are the same
        dim = len(x[0][0])
        batch_size = len(x)
        for i in range(batch_size):
            x[i][0] = edge_to_vertex(x[i][0], dim)
        return x


class convSym(nn.Module):
    # deletes symmetric part of all matrices by only keeping lower triangular matrix
    def __init__(self, channels):
        super(convSym, self).__init__()
        self.channels = channels

    def forward(self, x):
        # shape of x is [batch_size, dim, dim, dim]
        # iterate batch
        for i in range(len(x)):
            # iterate channels
            for j in range(self.channels):
                x[i][j] = x[i][j].tril()
        return x

class convEdge(nn.Module):
    # edge to edge filter layer
    def __init__(self, dim):
        super(convEdge, self).__init__()
        self.dim = dim

    def forward(self, x):
        # shape of x is [batch_size, dim, dim]
        for i in range(len(x)):
            x[i] = edge_to_edge(self.dim, x[i])
        return x.view(-1, 1, self.dim, self.dim).float()


class Net(nn.Module):
    """
    Class to define fully connected layer NN
    """
    def __init__(self, dim, batch_size):
        super(Net, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.convEdge = convEdge(self.dim)
        self.convSym = convSym(self.dim)
        self.convVer = convVert()
        self.fc1 = nn.Linear(dim * self.batch_size, 4)
        self.fc2 = nn.Linear(4, 2)

        self.cnn = nn.Sequential(
            self.convEdge,
            nn.Conv2d(1, 10, 1),
            nn.ReLU(),
            self.convSym,
            nn.Conv2d(10, 1, 3, 1, 1),
            nn.ReLU(),
            self.convVer,
        )

    def forward(self, x):
        # x = torch.from_numpy(self.convEdge(x)).double()
        # x = Variable(torch.from_numpy(x))
        x = self.cnn(x).view(10 * self.batch_size)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        return x

    def backprop(self, data, loss_func, epoch, optimizer, display_epoch, display_batch):
        total_loss = 0
        correct = 0
        dataset_len = len(data.train_loader.dataset)
        for batch_id, (x, y) in enumerate(data.train_loader):
            batch_size = len(y)
            output = self(x.view(-1, 10, 10).float())
            # figure out what to do here

            optimizer.zero_grad()
            output = output.view(-1, 2)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            # calculate total correct predictions in current batch
            predict = torch.max(output.data, 1)[1]
            correct += (predict == y).sum()
            total_loss += loss.item()
            if batch_id % display_batch == 0 and epoch % display_epoch == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_id * batch_size, dataset_len,
                           100 * batch_id * batch_size / dataset_len, loss.data.item()))


    def test(self, X_test, Y_test, loss):
        self.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test)
            targets = torch.from_numpy(Y_test)
            output = self.forward(inputs)
            cross_val = loss(output, targets.reshape(-1))
        return cross_val.item(), output, targets

if __name__ == '__main__':
    convEdge = convEdge(10)

[[-0.6864,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,0.0000,  0.0000],
[-0.6864, -0.6864,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000],
[-4.4750, -0.6864, -0.6864,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,0.0000,  0.0000],
[-4.9486, -0.6864, -4.9486, -0.6864,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000],
[-5.8958, -4.4750, -5.8958, -6.3694, -0.6864,  0.0000,  0.0000,  0.0000,0.0000,  0.0000],
[-4.4750, -3.0543, -4.4750, -0.6864, -5.8958, -0.6864,  0.0000,  0.0000,0.0000,  0.0000],
[-0.6864, -0.6864, -4.0015, -0.6864, -5.4222, -0.6864, -0.6864,  0.0000,0.0000,  0.0000],
[-0.6864, -0.6864, -0.6864, -4.4750, -0.6864, -0.6864, -3.5279, -0.6864,0.0000,  0.0000],
[-0.6864, -0.6864, -0.6864, -4.4750, -5.4222, -0.6864, -0.6864, -3.5279,-0.6864,  0.0000],
[-5.4222, -0.6864, -0.6864, -5.8958, -6.8429, -5.4222, -4.9486, -4.9486,-4.9486, -0.6864]]

[[-0.6864, -0.6864, -4.4750, -4.9486, -5.8958, -4.4750, -0.6864, -0.6864,-0.6864, -5.4222],
[-0.6864, -0.6864, -0.6864, -0.6864, -4.4750, -3.0543, -0.6864, -0.6864,-0.6864, -0.6864],
[-4.4750, -0.6864, -0.6864, -4.9486, -5.8958, -4.4750, -4.0015, -0.6864,-0.6864, -0.6864],
[-4.9486, -0.6864, -4.9486, -0.6864, -6.3694, -0.6864, -0.6864, -4.4750,-4.4750, -5.8958],
[-5.8958, -4.4750, -5.8958, -6.3694, -0.6864, -5.8958, -5.4222, -0.6864, -5.4222, -6.8429],
[-4.4750, -3.0543, -4.4750, -0.6864, -5.8958, -0.6864, -0.6864, -0.6864, -0.6864, -5.4222],
[-0.6864, -0.6864, -4.0015, -0.6864, -5.4222, -0.6864, -0.6864, -3.5279,-0.6864, -4.9486],
[-0.6864, -0.6864, -0.6864, -4.4750, -0.6864, -0.6864, -3.5279, -0.6864,-3.5279, -4.9486],
[-0.6864, -0.6864, -0.6864, -4.4750, -5.4222, -0.6864, -0.6864, -3.5279,-0.6864, -4.9486],
[-5.4222, -0.6864, -0.6864, -5.8958, -6.8429, -5.4222, -4.9486, -4.9486,-4.9486, -0.6864]]