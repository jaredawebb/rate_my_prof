# Jared Webb

print("Importing Libraries")
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import GridSearchCV # Do this for now.  Once this is working, switch to random.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from time import time
import sys

# Import all regression models.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveRegressor

mod = int(sys.argv[1])

print("Loading Data")
# Load Data sets.
train = './sample/sample_train.csv'
test = './sample/sample_test.csv'

comments_df = pd.read_csv(train, usecols=['comments'])
category_df = pd.read_csv(train)
category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
category_df.fillna(-1, inplace=True)
quality_df = pd.read_csv(train, usecols=['quality'])

test_comments_df = pd.read_csv(test, usecols=['comments'])
test_category_df = pd.read_csv(test)
test_category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
test_category_df.fillna(-1, inplace=True)
test_quality_df = pd.read_csv(test, usecols=['quality'])

test_ids = pd.read_csv(test, usecols=['id'])

print("Transforming Data\n")
# Transform comment data
# tfidfvectorizer = TfidfVectorizer(min_df=120, ngram_range=(1,2))
tfidfvectorizer = TfidfVectorizer(min_df=120, ngram_range=(1,2))

comm_train = tfidfvectorizer.fit_transform(comments_df['comments'].fillna(''))
comm_test = tfidfvectorizer.transform(test_comments_df['comments'].fillna(''))

# Stack feature and comment data, train_test_split
feat_train = category_df.values
Xtrain = sp.hstack((sp.csr_matrix(category_df.values), comm_train))
Xtest = sp.hstack((sp.csr_matrix(test_category_df.values), comm_test))
Ytrain = np.ravel(quality_df['quality'])
Ytest = np.ravel(test_quality_df['quality'])
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain, Ytrain,test_size=.25, random_state=0)

ids = test_ids.id 
# Train Models.
params = {}

def run_grid_search(m, parameters, params, name, Xtrain, Ytrain, Xtest, Ytest):
	print('=' * 80)
	print("Training %s Model" % name)
	print('=' * 80)
	t0 = time()

	clf = GridSearchCV(m, parameters, cv=3, n_jobs=8, verbose=3, error_score=0)
	clf.fit(Xtrain, Ytrain)
	Yhat = clf.predict(Xtest)
	print("\tDone in %1.2f seconds" % float(time() - t0))
	print("\tScore: %1.2f\n" % mse(Yhat, Ytest))

	print("Best Parameters" + str(clf.best_params_))
	print("Writing Solution")
	submit = pd.DataFrame(data={'id': ids, 'quality': Yhat})
	submit.to_csv('./submissions/'+name+'.csv', index = False)

if mod == 0:
	parameters = {'alpha':np.power(10.0, [2,3])}
	m = Ridge()
	run_grid_search(m, parameters, params, 'Test', Xtrain, Ytrain, Xtest, Ytest)
	print("Static Fire Successful")

if mod == 1:
	parameters = {'alpha':np.power(10.0, np.arange(-4,-2)), 'normalize':[True, False],
					'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag'],
					}

	m = Ridge()
	run_grid_search(m, parameters, params, 'Ridge', Xtrain, Ytrain, Xtest, Ytest)

if mod == 2:
	parameters = {'alpha':np.power(10.0, np.arange(-4,5)), 'normalize':[True, False],
					'positive':[True, False],
					'selection':['random', 'cyclic']}

	m = Lasso()
	run_grid_search(m, parameters, params, 'Lasso', Xtrain, Ytrain, Xtest, Ytest)

if mod == 3:
	parameters = {'alpha':np.power(10.0, np.arange(-4,5)), 'l1_ratio':np.arange(0,1.05,0.05),
					'normalize':[True, False], 'positive':[True, False],
					'selection':['random', 'cyclic']}
	m = ElasticNet()
	run_grid_search(m, parameters, params, 'ElasticNet', Xtrain, Ytrain, Xtest, Ytest)

if mod == 4:
	parameters = {'alpha':np.power(10.0, np.arange(-4,5)), 'normalize':[True, False],
					'selection':['random', 'cyclic']}

	m = MultiTaskLasso()
	run_grid_search(m, parameters, params, 'MultiTaskLasso', Xtrain, Ytrain, Xtest, Ytest)

if mod == 5:
	# Lars
	parameters = {'normalize':[True, False], 'n_nonzero_coefs':[10,25,50,150,500,np.inf]}
	m = Lars()
	run_grid_search(m, parameters, params, 'Lars', Xtrain, Ytrain, Xtest, Ytest)

if mod == 6:
	# LassoLars
	parameters = {'alpha':np.power(10.0, np.arange(-4,5)), 'positive':[True, False], 'normalize':[True, False]}
	m = LassoLars()
	run_grid_search(m, parameters, params, 'LarsLasso', Xtrain, Ytrain, Xtest, Ytest)

if mod == 7:
	# BayesianRidge
	parameters = {'alpha_1':np.power(10.0, np.arange(-9,-3)), 'alpha_2':np.power(10.0, np.arange(-9,-3)), 
					'lambda_1':np.power(10.0, np.arange(-9,-3)), 'lambda_2':np.power(10.0, np.arange(-9,-3)),
					'compute_score':[True,False], 'normalize':[True,False]}
	m = BayesianRidge()
	run_grid_search(m, parameters, params, 'BayesianRidge', Xtrain, Ytrain, Xtest, Ytest)

if mod == 8:
	# ARDRegression
	parameters = {'alpha_1':np.power(10.0, np.arange(-9,-3)), 'alpha_2':np.power(10.0, np.arange(-9,-3)), 
					'lambda_1':np.power(10.0, np.arange(-9,-3)), 'lambda_2':np.power(10.0, np.arange(-9,-3)),
					'compute_score':[True,False], 'normalize':[True,False], 'threshold_lambda':np.power(10.0, np.arange(2,6))}
	m = ARDRegression()
	run_grid_search(m, parameters, params, 'ARDRegressor', Xtrain, Ytrain, Xtest, Ytest)

if mod == 9:
	# LogisticRegression
	parameters = {'penalty':['l1', 'l2'], 'C':np.power(10.0, np.arange(-4,-1)), 'class_weight':[None,'balanced'],
					'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag'], 'multi_class':['ovr', 'multinomial']}
	m = LogisticRegression()
	run_grid_search(m, parameters, params, 'LogisticRegression', Xtrain, Ytrain, Xtest, Ytest)

if mod == 10:
	# SGDRegressor
	parameters = {'loss':['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'penalty':[None, 'l1', 'l2', 'elasticnet'],
					'alpha':np.power(10.0, np.arange(-4,5)), 'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5], 'shuffle':[True, False], 'epsilon':[0.00001],
					'learning_rate':['constant', 'invscaling'], 'eta0':[0.0001, 0.001, 0.01, 0.1]}
	m = SGDRegressor()
	run_grid_search(m, parameters, params, 'SGDRegressor', Xtrain, Ytrain, Xtest, Ytest)

if mod == 11:
	# Perceptron
	parameters = {'penalty':['l2', 'l1', 'elasticnet'], 'alpha':np.power(10.0, np.arange(-4,5)), 'shuffle':[True, False], 'eta0':np.power(10.0, np.arange(-1,5)),
					'class_weight':['balanced', 'none']}
	m = Perceptron()
	run_grid_search(m, parameters, params, 'Perceptron', Xtrain, Ytrain, Xtest, Ytest)

if mod == 12:
	# PassiveAgressiveLearner
	parameters = {'C':np.power(10.0, np.arange(-3,3)), 'epsilon':np.power(10.0, np.arange(-4,-1)), 'shuffle':[True, False]}

	m = PassiveAgressiveLearner()
	run_grid_search(m, parameters, params, 'PassiveAgressiveLearner', Xtrain, Ytrain, Xtest, Ytest)

if mod == 13:
	print('=' * 80)
	print("Training Least Squares Model")
	print('=' * 80)
	t0 = time()

	m = LinearRegression()
	m.fit(Xtrain, Ytrain)
	Yhat = m.predict(Xtest)
	print("Done in %1.2f seconds" % float(time() - t0))
	print("Score: %1.2f" % mse(Yhat, Ytest))
	print("Writing Solution")
	submit = pd.DataFrame(data={'id': ids, 'quality': Yhat})
	submit.to_csv('./submissions/LinearRegression.csv', index = False)


