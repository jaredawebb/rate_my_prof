import numpy as np
import sys
import pandas as pd

def calc_rmse(submission_file, evaluation_file = './sample/sample_test_y.csv'):
	'''
	A simple function to calculate the Root Mean Squared Error.
	I'm sure that there is a python module that does this...probably sklearn has one.
	It was easier to just write one though.
	'''

	sub = pd.read_csv(submission_file)
	eva = pd.read_csv(evaluation_file)

	x = np.array(sub)
	y = np.array(eva)

	print("RMSE: " + str(np.sqrt(np.sum((x-y)**2)/len(x))))
