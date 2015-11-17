# rate_my_prof
Code for the MDST inclass kaggle competition.

This repository contains several scripts for analyzing the MDST rate my professor data.
It also contains some empty directories that are used by the scripts to store data and submissions.


Scripts and explanations:

1. create_sample.py
This script needs the cleaned data from MDST.  This can be created using the code from the MDST repository.
The data should be stored in the ./data/ directory.
The sampled data will be saved to the ./sample/ directory.

2. feats_regr.py
This script runs an XGboost and RandomForest classifier on the non-comment data.
The predicted scores will be stored in the ./submissions/ directory.

3. lda_comments.py
This script run LDA on the comment data and then uses the calculated topic distributions as features
for a ridge regression model.
The predicted scores will be stored in the ./submissions/ directory.

4. rmse.py
A simple script that calculates the root mean squared error.

Other files:

1.  notes
Explanations of the feature names for now.  Can be anything really.

2. stop_words.txt
A small file of words that should be ignored when running LDA.  This list should probably
grow.

