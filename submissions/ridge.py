#
# MDST Ratings Analysis Challenge
# Starter code & ridge regression baseline
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
import sklearn.linear_model
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
import numpy as np
import re

np.random.seed(0)

# Load in the data - pandas DataFrame objects

rats_tr = pd.read_csv('sample/sample_train.csv')
rats_te = pd.read_csv('data/sampe_test.csv')

# Let's take a look at the feature

print("Training set columns:")
print(rats_tr.columns.tolist())

print("Training set size:", rats_tr.shape)

print("Test set columns:")
print(rats_te.columns.tolist())

print("Test set size:", rats_te.shape)

# We're going predict 'quality' from the 'comments'.
# Let's look at a few random comments and the quality rating
inds = np.random.choice(range(rats_tr.shape[0]),size=10,replace=False)
print(rats_tr.loc[inds, ['comments', 'quality']])

# Construct bigram representation
count_vect = CountVectorizer(min_df=120,stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2))


# "Fit" the transformation on the training set and apply to test
Xtrain = count_vect.fit_transform(rats_tr.comments.fillna(''))
Xtest = count_vect.transform(rats_te.comments.fillna(''))






# Now let's train a model

Ytrain = np.ravel(rats_tr.quality)
ybar = np.mean(Ytrain)

cl = sklearn.linear_model.Ridge()
cl.fit(Xtrain,np.array(Ytrain))
Yhat = cl.predict(Xtest)

df = pd.DataFrame(data={'words':count_vect.get_feature_names(),
                        'coef':cl.coef_.flatten()
                    })
df.sort('coef',ascending=False,inplace=True)

print("Ridge Coefficients")
print("Most positive:")
print(df[0:30])
print("Most negative")
print(df[-30:])


# Save results in kaggle format
submit = pd.DataFrame(data={'id': rats_te.id, 'quality': Yhat})
submit.to_csv('ridge_submit.csv', index = False)





# How do we make a prediction?
# What does the bigram representation look like?
print(rats_tr.loc[0, 'comments'])

bg = pd.DataFrame(data={'x': np.ravel(Xtrain[0, :].todense()),
                        'words': count_vect.get_feature_names()
                    })
bg.sort('x', ascending=False, inplace = True)
bg = bg.merge(df, on='words')

print("Bigram representation:")
print(bg[0:30])
