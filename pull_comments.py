import pandas as pd
import numpy as np

# This script will create a csv with just the ids and comments.

datafile = './sample/sample_train.csv'

df = pd.read_csv(datafile, usecols=['id', 'comments'])

df.to_csv('./sample/sample_comments.csv')


