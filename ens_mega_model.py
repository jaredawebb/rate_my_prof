# Jared Webb

print("Importing Libraries")
import numpy as np
import scipy.sparse as sp
import pandas as pd
import textblob
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import GridSearchCV # Do this for now.  Once this is working, switch to random.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse
from time import time
import sys

# Import all regression models.

from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

mod = int(sys.argv[1])

print("Loading Data")
# Load Data sets.

train = './sample/sample_train.csv'
test = './sample/sample_test.csv'
'''
train = './data/newtrain.csv'
test = './data/newtest.csv'
'''
train_sent = './data/newtrain_sentiments.csv'
test_sent = './data/newtest_sentiments.csv'

comments_df = pd.read_csv(train, usecols=['comments'])
category_df = pd.read_csv(train)
category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
category_df.fillna(-1, inplace=True)
quality_df = pd.read_csv(train, usecols=['quality'])
sentiment_df = pd.read_csv(train_sent)


test_comments_df = pd.read_csv(test, usecols=['comments'])
test_category_df = pd.read_csv(test)
# test_category_df.drop(['comments', 'id', 'tid'], axis=1, inplace=True)
test_category_df.drop(['quality', 'clarity', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
test_category_df.fillna(-1, inplace=True)
test_quality_df = pd.read_csv(test, usecols=['quality'])
test_sentiment_df = pd.read_csv(test_sent)

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
#Xtrain = sp.hstack((Xtrain, sp.csr_matrix(sentiment_df[['polarity', 'subjectivity']].values)))

Xtest = sp.hstack((sp.csr_matrix(test_category_df.values), comm_test))
#print(Xtest.shape, test_sentiment_df.shape)
#Xtest = sp.hstack((Xtest, sp.csr_matrix(test_sentiment_df[['polarity', 'subjectivity']].values)))

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

	clf = GridSearchCV(m, parameters, cv=3, n_jobs=4, verbose=3, error_score=0)
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
	m = KernelRidge()
	run_grid_search(m, parameters, params, 'Test', Xtrain, Ytrain, Xtest, Ytest)
	print("Static Fire Successful")

if mod == 1:
	parameters = {'alpha':np.power(10.0, np.arange(-4,-2)),'kernel':['linear', 'RBF', 'laplacian', 'sigmoid']}

	m = KernelRidge()
	run_grid_search(m, parameters, params, 'KernelRidge', Xtrain, Ytrain, Xtest, Ytest)

if mod == 2:
	parameters = {'n_neighbors':np.arange(3,7),'weights':['uniform', 'distance'], 'p':np.arange(1,4)} 

	m = KNeighborsRegressor()
	run_grid_search(m, parameters, params, 'NearestNeighbor', Xtrain, Ytrain, Xtest, Ytest)

if mod == 3:
	parameters = {'n_components':[2,3,4], 'tol':np.power(10.0, np.arange(-7,-4))}

	m = PLSRegression()
	run_grid_search(m, parameters, params, 'PLSRegression', Xtrain, Ytrain, Xtest, Ytest)

if mod == 4:
	parameters = {'splitter':['best', 'random'], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[None, 8, 9, 10]}

	m = DecisionTreeRegressor()
	run_grid_search(m, parameters, params, 'DecisionTree', Xtrain, Ytrain, Xtest, Ytest)

if mod == 5:
	# Lars
	parameters = {'n_estimators':[100, 150, 200], 'bootstrap':[True, False], 
					'bootstrap_features':[True, False], 'max_features':[0.5, 0.75, 1.0]}
	m = BaggingRegressor()
	run_grid_search(m, parameters, params, 'Bagging', Xtrain, Ytrain, Xtest, Ytest)

if mod == 6:
	# LassoLars
	parameters = {'n_estimators':[100, 150, 200, 500], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[8, 10, 12],
					'bootstrap':[True, False]}
	m = RandomForestRegressor()
	run_grid_search(m, parameters, params, 'RandomForest', Xtrain, Ytrain, Xtest, Ytest)

if mod == 7:
	# BayesianRidge
	parameters = {'n_estimators':[100, 150, 200, 500], 'max_features':['auto', 'sqrt', 'log2'], 'max_depth':[8, 10, 12],
					'bootstrap':[True, False]}
	
	m = ExtraTreesRegressor()
	run_grid_search(m, parameters, params, 'ExtraTrees', Xtrain, Ytrain, Xtest.toarray(), Ytest)

if mod == 8:
	# ARDRegression
	parameters = {'n_estimators':[100, 150, 200, 500], 'learning_rate':[1.0, 10.0, 0.1], 'loss':['linear', 'square', 'exponential']}
	m = AdaBoostRegressor()
	run_grid_search(m, parameters, params, 'AdaBoost', Xtrain.toarray(), Ytrain, Xtest.toarray(), Ytest)

if mod == 9:
	# LogisticRegression
	parameters = {'loss':['ls', 'lad', 'huber', 'quantile'], 'learning_rate':[0.1, 1.0, 10.0], 'n_estimators':[100, 500, 1000]}
	m = GradientBoostingRegressor()
	run_grid_search(m, parameters, params, 'GradientBoosting', Xtrain, Ytrain, Xtest, Ytest)

'''
if mod == 10:
	parameters = {'max_depth':np.arange(9,12), 'n_estimators':[3000], 'learning_rate':np.power(10.0, np.arange(-3,-1)), 'subsample':np.arange(0.25,1.25,0.25)}#,
#				'colsample_bytree':np.arange(0,1.25, .25), 'gamma':np.arange(0,1.25, .25), 'base_score':[.5, .6, .7], 'seed':[42]}
	m = xgb.XGBRegressor()
	#Xtrain = xgb.DMatrix(Xtrain)
	#Xtest = xgb.DMatrix(Xtest)
	run_grid_search(m, parameters, params, 'XGBoost', Xtrain, Ytrain, Xtest, Ytest)	
'''
