import pandas as pd
import numpy as np

num_vertices = [6, 7, 8, 9]
for vert in num_vertices:
    distant1_df = pd.read_csv(f'../data/graphs1_{vert}.csv', header=None, sep=' ')
    distant2_df = pd.read_csv(f'../data/graphs_{vert}.csv', header=None, sep=' ')
    random_df = pd.read_csv(f'../data/graphs_{vert}_rand.csv', header=None, sep=' ')
    combined_df = pd.concat([distant1_df, distant2_df, random_df], sort=False)

    index_classic_err = combined_df[(combined_df.iloc[:, -2] == -1.0)].index
    combined_df.drop(index_classic_err, inplace=True)

    before_duplicate_removal = len(combined_df.index)

    # remove all duplicates based on graph and start/end point
    combined_df = combined_df.drop_duplicates(subset=combined_df.columns[:-2])

    quantum_df = combined_df[(combined_df.iloc[:, -2] > combined_df.iloc[:, -1])]
    # quantum_df.reset_index(drop=True, inplace=True)

    len_quant = len(quantum_df.index)
    classical_df = combined_df[(combined_df.iloc[:, -2] < combined_df.iloc[:, -1])].sample(n=1000-len_quant)
    new_df = quantum_df.append(classical_df)
    new_df.to_csv(f'../data/graphs_{vert}_combined.csv', index=False, header=False, sep=' ')







