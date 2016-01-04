import pandas as pd
import numpy as np

# This script will generate a pull a random set of rows from the training set for testing.

datafile = './data/newtrain.csv'

df = pd.read_csv(datafile)
num_rows = len(df)

# Set size of sample here.
sample_size = int(num_rows * (0.75))

# Choose the rows to put in train at random.
rows = np.random.choice(df.index.values, sample_size)
sampled_df = df.ix[rows]

# The remaining rows go in test.
other_df = df.drop(rows)

# Save the files.
sampled_df.to_csv('./sample/sample_train.csv', index=False)
other_df.to_csv('./sample/sample_test.csv', index=False)
