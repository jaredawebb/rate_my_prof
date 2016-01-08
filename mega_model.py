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

# Import all regression models.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

print("Loading Data")
# Load Data sets.
comments_df = pd.read_csv('./sample/sample_train.csv', usecols=['comments'])
category_df = pd.read_csv('./sample/sample_train.csv')
category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
category_df.fillna(-1, inplace=True)
quality_df = pd.read_csv('./sample/sample_train.csv', usecols=['quality'])

test_comments_df = pd.read_csv('./sample/sample_test.csv', usecols=['comments'])
test_category_df = pd.read_csv('./sample/sample_test.csv')
test_category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
test_category_df.fillna(-1, inplace=True)
test_quality_df = pd.read_csv('./sample/sample_test.csv', usecols=['quality'])

print("Transforming Data\n")
# Transform comment data
tfidfvectorizer = TfidfVectorizer(min_df=1, max_df=100, ngram_range=(1,2))
comm_train = tfidfvectorizer.fit_transform(comments_df['comments'].fillna(''))
comm_test = tfidfvectorizer.transform(test_comments_df['comments'].fillna(''))

# Stack feature and comment data, train_test_split
feat_train = category_df.values
Xtrain = sp.hstack((sp.csr_matrix(category_df.values), comm_train))
Xtest = sp.hstack((sp.csr_matrix(test_category_df.values), comm_test))
Ytrain = np.ravel(quality_df['quality'])
Ytest = np.ravel(test_quality_df['quality'])
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain, Ytrain,test_size=.25, random_state=0)

# Train Models.

print('=' * 80)
print("Training Least Squares Model")
print('=' * 80)
t0 = time()

m = LinearRegression()
m.fit(Xtrain, Ytrain)
yhat_lr = m.predict(Xtest)
print("Done in %1.2f seconds" % float(time() - t0))
print("Score: %1.2f" % mse(yhat_lr, Ytest))

params = {}

def run_grid_search(m, parameters, params, name, comm_train, Ytrain, comm_test, Ytest):
	print('=' * 80)
	print("Training %s Model" % name)
	print('=' * 80)
	t0 = time()

	clf = GridSearchCV(m, parameters, cv=5, n_jobs=-1, error_score=0)
	clf.fit(Xtrain, Ytrain)
	yhat_ri = clf.predict(Xtest)
	params[name] = clf.best_params_
	print("Done in %1.2f seconds" % float(time() - t0))
	print("Score: %1.2f" % mse(yhat_lr, Ytest))


parameters = {'alpha': np.power(10, np.arange(-4,5)), 'normalize':[True, False],
				'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag'],
				'tol': np.power(10, np.arange(-7,-1))}

m = Ridge()
run_grid_search(m, parameters, params, 'Ridge', comm_train, Ytrain, comm_test, Ytest)


parameters = {'alpha':np.power(10, np.arange(-4,5)), 'normalize':[True, False],
				'tol': np.power(10, nparange(-7,-1)), 'positive':[True, False],
				'selection':['random', 'cyclic']}

m = Lasso()
run_grid_search(m, parameters, params, 'Lasso', Xtrain, Ytrain, Xtest, Ytest)

parameters = {'alpha':np.power(10, np.arange(-4,5)), l1_ratio:np.arange(0,1.05,0.05),
				'normalize':[True, False], 'tol':np.power(10, np.arange(-7,-1)), 'positive':[True, False],
				'selection':['random', 'cylcic']}
m = ElasticNet()
run_grid_search(m, parameters, params, 'ElasticNet', Xtrain, Ytain, Xtest, Ytest)









