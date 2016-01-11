import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PassiveAggressiveLearner

print("Loading Data")
train = './data/newtrain.csv'
test = './data/newtest.csv'

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

print("Transforming Data")
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

print("Training Models")

m1 = Ridge(normalize=True, alpha=0.001, solver='auto')
m2 = Lasso(normalize=False, alpha=0.0001, solver='cyclic',positive=False)
m3 = ElasticNet(normalize=False, alpha=0.0001,positive=False, l1_ratio = 0.2)
m4 = PassiveAgressiveLearner(epsilon=0.001, C=100, shuffle=True)
m5 = LinearRegression()

m1.fit(Xtrain, Ytrain)
m2.fit(Xtrain, Ytrain)
m3.fit(Xtrain, Ytrain)
m4.fit(Xtrain, Ytrain)
m5.fit(Xtrain, Ytrain)

models = [m1, m2, m3, m4, m5]

X = np.zeros(len(y1), 5)
for i in range(len(models)):
	y = models[i].predict(Xtest)
	X[:,i] = np.ravel(y)


m = LinearRegression
Xtr, Xte, Ytr, Yte = train_test_split(X, Ytrain, test_size=.25)
m.fit(Xtr,Ytr)
yhat = m.predict(Xte)
print("Score: " + str(mse(yhat, Yte)))
yfinal = m.predict(X)

print("Writing Solution")
submit = pd.DataFrame(data={'id': ids, 'quality': Yhat})
submit.to_csv('./submissions/ensemble.csv', index = False)


