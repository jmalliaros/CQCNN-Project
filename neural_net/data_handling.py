import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data


class AdjData:
    def __init__(self, csv_path, test_size, batch_size=1):
        """
        :param test_size: specifies the size of the test set (decimal)

        - reads csv data into data frame and splits into data and labels
        - uses scikit learn train_test_split to split into train and test (n=3000) sets
        """
        adj_df = pd.read_csv(csv_path, header=None, sep=' ')
        q_steps = adj_df.iloc[:, -1]
        adj_df = adj_df.iloc[:, :-1]
        c_steps = adj_df.iloc[:, -1]
        adj_df = adj_df.iloc[:, :-1]
        # drop start and end pos
        adj_df = adj_df.iloc[:, :-1]
        adj_df = adj_df.iloc[:, :-1]
        # faster on classical = 1, quantum = 0
        labels = pd.Series([1 if q_steps[i] > c_steps[i] else 0 for i in range(len(q_steps))])

        # print(adj_df)
        x_train, x_test, y_train, y_test = train_test_split(
                                                                adj_df,
                                                                labels,
                                                                test_size=test_size)
        # convert to tensor
        x_train = torch.from_numpy(x_train.values)
        y_train = torch.from_numpy(y_train.values)
        x_test = torch.from_numpy(x_test.values)
        y_test = torch.from_numpy(y_test.values)

        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        self.train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

        # print('ok')

if __name__ == '__main__':
    adj = AdjData('C:/Users/Haoming/phys490/graph-generator/data/graphs_10.csv', .1)