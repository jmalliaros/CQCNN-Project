import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class AdjData():
    def __init__(self, test_size):
        """
        :param test_size: specifies the size of the test set (decimal)

        - reads csv data into data frame and splits into data and labels
        - uses scikit learn train_test_split to split into train and test (n=3000) sets
        """
        adj_df = pd.read_csv('data/graphs_10.csv', header=None, names=['data'])
        adj_df['quant_steps'] = pd.to_numeric(adj_df['data'].str[-1])
        adj_df['class_steps'] = pd.to_numeric(adj_df['data'].str[-1])
        adj_df['end_pos'] = pd.to_numeric(adj_df['data'].str[-1])
        adj_df['start_steps'] = pd.to_numeric(adj_df['data'].str[-1])


        adj_df['data'] = adj_df['data'].apply(lambda d: list(map(int, d.split()))[:-4])

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                                                                adj_df['data'],
                                                                adj_df['class_steps'],
                                                                test_size=test_size)
