import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, PassiveAggressiveRegressor

print("Loading Data")
train = './data/newtrain.csv'
train_sent = './data/newtrain_sentiment.csv'
test = './data/newtest.csv'
test_sent = './data/newtest_sentiment.csv'

comments_df = pd.read_csv(train, usecols=['comments'])
category_df = pd.read_csv(train)
category_df.drop(['quality', 'clarity','easiness', 'helpfulness', 'comments', 'id','tid'], axis=1, inplace=True)
category_df.fillna(-1, inplace=True)
quality_df = pd.read_csv(train, usecols=['quality'])
sent_df = pd.read_csv(train_sent)


test_comments_df = pd.read_csv(test, usecols=['comments'])
test_category_df = pd.read_csv(test)
test_category_df.drop(['id','tid', 'comments'], axis=1, inplace=True)
test_category_df.fillna(-1, inplace=True)
#test_quality_df = pd.read_csv(test, usecols=['quality'])
test_sent_df = pd.read_csv(test_sent)

test_ids = pd.read_csv(test, usecols=['id'])

print("Transforming Data")
tfidfvectorizer = TfidfVectorizer(min_df=120, ngram_range=(1,2))

comm_train = tfidfvectorizer.fit_transform(comments_df['comments'].fillna(''))
comm_test = tfidfvectorizer.transform(test_comments_df['comments'].fillna(''))

# Stack feature and comment data, train_test_split
feat_train = category_df.values
Xtrain = sp.hstack((sp.coo_matrix(category_df.values), comm_train))
Xtrain = sp.hstack((Xtrain, sp.csr_matrix(sent_df[['polarity', 'subjectivity']].values)))

Xtest = sp.hstack((sp.coo_matrix(test_category_df.values), comm_test))
Xtest = sp.hstack((Xtest, sp.csr_matrix(test_sent_df[['polarity', 'subjectivity']].values)))
Ytrain = np.ravel(quality_df['quality'])
#Ytest = np.ravel(test_quality_df['quality'])
Xtr, Xte, Ytr, Yte = train_test_split(Xtrain, Ytrain,test_size=.25, random_state=0)

ids = test_ids.id

print("Training Models")

m1 = Ridge(normalize=True, alpha=0.001, solver='auto')
m2 = Lasso(normalize=False, alpha=0.0001, selection='cyclic',positive=False)
m3 = ElasticNet(normalize=False, alpha=0.0001,positive=False, l1_ratio = 0.2)
m4 = PassiveAggressiveRegressor(epsilon=0.001, C=100, shuffle=True)
m5 = LinearRegression()

m1.fit(Xtrain, Ytrain)
print("Model 1 Finished")
m2.fit(Xtrain, Ytrain)
print("Model 2 Finished")
m3.fit(Xtrain, Ytrain)
print("Model 3 Finished")
m4.fit(Xtrain, Ytrain)
print("Model 4 Finished")
m5.fit(Xtrain, Ytrain)
print("Model 5 Finished")


models = [m1, m2, m3, m4, m5]

X = np.zeros((Xtest.shape[0], 5))
Xt = np.zeros((Xtr.shape[0], 5))
for i in range(len(models)):
	y = models[i].predict(Xtest)
	X[:,i] = np.ravel(y)
	Xt[:,i] = models[i].predict(Xtr)
	submit = pd.DataFrame(data={'id': ids, 'quality': Yhat})
	submit.to_csv('./submissions/ensemble_m_'+str(i)+'.csv', index = False)


Xtr, Xte, Ytr, Yte = train_test_split(Xt, Ytr, test_size=.15)
m = LinearRegression()
m.fit(Xtr,Ytr)
yhat = m.predict(Xte)
print("Score: " + str(mse(yhat, Yte)))
yfinal = m.predict(X)

print("Writing Solution")
submit = pd.DataFrame(data={'id': ids, 'quality': yfinal})
submit.to_csv('./submissions/ensemble.csv', index = False)


