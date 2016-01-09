#
# MDST Ratings Analysis Challenge
# Model selection with Cross Validation
#
# Jonathan Stroud
#
#
# Prerequisites:
#
# numpy
# pandas
# sklearn
#

import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.sparse import vstack

from sklearn import linear_model, cross_validation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Load in the data - pandas DataFrame objects
rats_tr = pd.read_csv('data/newtrain.csv')
rats_te = pd.read_csv('data/newtest.csv')

# Construct sparse features
feats = rats_tr[['profgender', 'profhotness', 'online']].fillna(-1)
sparse_feats = feats.to_sparse()
del feats

# Construct bigram representation
count_vect = CountVectorizer(min_df=10,stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2))

# Calculate sentiments
sentiments = np.zeros(len(rats_te))
count = 0
for comment in rats_te['comments'].tolist():
	tb_comment = TextBlob(str(comment))
	sentiments[count] = tb_comment.sentiment.polarity
	count += 1

# "Fit" the transformation on the training set and apply to test
Xtrain = vstack(sparse_feats, count_vect.fit_transform(rats_tr.comments.fillna('')))
Xtest = count_vect.transform(rats_te.comments.fillna(''))

Ytrain = np.ravel(rats_tr.quality)

# Select alpha with a validation set
Xtr, Xval, Ytr, Yval = cross_validation.train_test_split(
    Xtrain,
    Ytrain,
    test_size = 0.25,
    random_state = 0)

# Define window to search for alpha
alphas = np.power(10.0, np.arange(-2, 8))

# Store MSEs here for plotting
mseTr = np.zeros((len(alphas),))
mseVal = np.zeros((len(alphas),))

# Search for lowest validation accuracy
for i in range(len(alphas)):
    print("alpha =", alphas[i])
    m = linear_model.Ridge(alpha = alphas[i]) 
    m.fit(Xtr, Ytr)
    YhatTr = m.predict(Xtr)
    YhatVal = m.predict(Xval)
    mseTr[i] = mean_squared_error(YhatTr, Ytr)
    mseVal[i] = mean_squared_error(YhatVal, Yval)
print(mseTr)
print(mseVal)

print(mseTr)
print(mseVal)
'''
import matplotlib.pyplot as plt
plt.semilogx(alphas, mseTr, hold=True)
plt.semilogx(alphas, mseVal)
plt.legend(['Training MSE', 'Validation MSE'])
plt.ylabel('MSE')
plt.xlabel('alpha')
plt.show()
'''
# Best performance at alpha = 100
# Train new model using all of the training data
m = linear_model.Ridge(alpha = 100)
m.fit(Xtrain, Ytrain)
Yhat = m.predict(Xtest)
'''
# SGD Classifier
m1 = linear_model.SGDRegressor(alpha=0.0001, average=False, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='huber', n_iter=50,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

m1.fit(Xtr, Ytr)
Yhat1 = m1.predict(Xval)
print("MSE: " + str(mean_squared_error(Yhat1, Yval)))

m1 = linear_model.SGDRegressor(alpha=0.0001, average=False, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='huber', n_iter=50,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

m1.fit(Xtrain, Ytrain)
Yhat1 = m1.predict(Xtest)
'''

# Fix Yhat. Try this first. Compare to normalizing
for i in range(len(Yhat)):
	if Yhat[i] > 10:
		Yhat[i] = 10
	if Yhat[i] < 2:
		Yhat[i] = 2
	if Yhat1[i] > 10:
		Yhat1[i] = 10
	if Yhat1[i] < 2:
		Yhat1[i] = 2

# Account for sentiment/prediction mismatch.
'''
count = 0
for i in range(len(Yhat)):
	if Yhat[i] > 6 and sentiments[i] < -0.5:
		Yhat[i] = (sentiments[i] + 1)*4 + 2


# Save results in kaggle format
submit = pd.DataFrame(data={'id': rats_te.id, 'quality': Yhat})
submit.to_csv('crossvalidation_submit.csv', index = False)

submit = pd.DataFrame(data={'id': rats_te.id, 'quality': Yhat1})
submit.to_csv('sgd.csv', index = False)

'''

# Other things to try:
# Add other features
# Other regularization types (lasso)
#     http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# Decision trees
#     http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# Random forests
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Boosting
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
