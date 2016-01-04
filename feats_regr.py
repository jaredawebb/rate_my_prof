import pandas as pd
import sklearn.linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np


# First choose whether you are doiing this on a sample
# or on the whole data set.

datafile1 = './sample/sample_train.csv'
datafile2 = './sample/sample_test.csv'
'''
datafile1 = './data/newtrain.csv'
datafile2 = './data/newtest.csv'
'''
# Prepare the data for xgboost and randomforest
print("Prepping Data")
Xtrain = pd.read_csv(datafile1)
Xtrain.drop(['comments', 'helpfulness', 'clarity', 'easiness', 'quality', 'id', 'tid'], axis=1, inplace=True)
#Xtrain.drop('quality', axis=1, inplace=True)
Xtrain.replace(to_replace='nan', value=0, inplace=True)
Ytrain = pd.read_csv(datafile1, usecols=['quality'])

# Convert to numpy arrays

Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

'''
cl2 = sklearn.linear_model.Ridge()
cl2.fit(Xtrain, Ytrain)
'''
# Train xgboost and randomforest

print("Training XGBoost")
gbm = xgb.XGBRegressor(max_depth=8, n_estimators=250, learning_rate=0.01,subsample = 0.7,colsample_bytree = 0.8)
gbm=gbm.fit(Xtrain,Ytrain, eval_metric = "auc")

print("Training RandomForest")
rf = RandomForestRegressor(n_estimators = 500, max_features = 'log2')
rf = rf.fit(Xtrain, np.ravel(Ytrain))

# Now prepare the test data.
Xtest = pd.read_csv(datafile2)
ids = Xtest['id']

Xtest.drop(['comments', 'id', 'tid'], axis=1, inplace=True)

# This makes it so that we can use newtest or sampletest in the same script.
toDrop = ['quality', 'helpfulness', 'clarity', 'easiness']
for thing in toDrop:
	if thing in Xtest.columns:
		Xtest.drop([thing], axis=1, inplace=True)

Xtest.replace(to_replace='nan', value=0, inplace=True)

Xtest = np.array(Xtest)

# Predict using the xgboost regressor.  Save in Kaggle format.
Yhat = gbm.predict(Xtest)
submit2 = pd.DataFrame(data={'id': ids, 'quality': np.ravel(Yhat)})
submit2.to_csv('./submissions/submit_xgb_r.csv', index = False)

# Predict using the random forest regressor.  Save in Kaggle format.
Yhat2 = rf.predict(Xtest)
submit2 = pd.DataFrame(data={'id': ids, 'quality': np.ravel(Yhat2)})
submit2.to_csv('./submissions/submit_rf_r.csv', index = False)
