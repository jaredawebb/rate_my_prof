
## Logistic Regression using Stochastic Gradient Descent
## with adapted learning rate ##
## modified by Arya Farahi
## http://www-personal.umich.edu/~aryaf/

# Giving due credit 
# classic tinrtgu's code
# https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
#modified by rcarson available as FTRL starter on kaggle code 
#https://www.kaggle.com/jiweiliu/springleaf-marketing-response/ftrl-starter-code
##############################################################################
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import numpy as np
import pandas as pd
import pickle
import rmse

# Given a list of words, remove any that are
# in a list of stop words.
def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train='./sample/sample_train.csv'
test='./sample/sample_test.csv'

train = './data/train.csv'
test = './data/test.csv'

submission = 'sgd_subm1.csv'  # path of to be outputted submission file

#stopwords = list(ENGLISH_STOP_WORDS)

with open('./stop_words.txt', 'r') as f:
    f1 = f.read()
    stopwords = f1.split('\n')[:-1]

print("Stop words : ", stopwords)

# B, model
alpha = .1 	# learning rate
beta = 1.		
L1 = 0.0     	# L1 regularization, larger value means more regularized
L2 = 0.01     	# L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 24             # number of weights to use
interaction = False   # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None  # data after date N (exclusive) are used as validation
holdout = 100  # use every N training instance for holdout validation

##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class gradient_descent(object):

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # G: squared sum of past gradients
        # w: weights
        self.w = [0.] * D  
        self.G = [0.] * D 

    def _indices(self, x):
        ''' A helper generator that yields the indices in x
            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''
        # model
        w = self.w	

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            wTx += w[i]

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.G: increase by squared gradient
                self.w: weights
        '''
        # parameters
        alpha = self.alpha
        L1 = self.L1
        L2 = self.L2

        # model
        w = self.w
        G = self.G

        # gradient under logloss
        g = p - y
        # update z and n
        for i in self._indices(x):
            G[i] += g*g
            #w[i] -= alpha*1/sqrt(n[i]) * (g - L2*w[i])
            ## Learning rate reducing as 1/sqrt(n_i) :
            ## ALso gives good performance but below code gives better results
            w[i] -= alpha/(beta+sqrt(G[i])) * (g - L2*w[i])
            ## Learning rate reducing as alpha/(beta + sqrt of sum of g_i)

        self.w = w
        self.G = G

def logloss(p, y):
    ''' FUNCTION: L2 Loss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''
    return (p - y)**2


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path), delimiter=',')):
      
        try:
            ID = row['id']
            del row['id']
        except:
            pass

        # normalized quality \in [0 - 1]
        try: 
           target='quality' 
           y = (float(row[target]) - 2.0) / 8.0; del row[target]
        except: 
           y = 0    

        # build x
        x = []
        value = row['comments']
        # one-hot encode everything with hash trick
        # extracting features from comments
        if value != 'No Comments':
           # removing stopwords (does not make it better though)
           value = removeStopwords(value.lower().split(), stopwords)
           #value = value.lower().split()
           for ivalue in value:
              index = abs(hash(ivalue)) % D
              x.append(index)
        else: 
           x.append(-1)
        
        #labels: 
        #id,tid,dept,date,forcredit,attendance,textbookuse,interest,
        #grade,tags,comments,helpcount,nothelpcount,online,profgender,
        #profhotness,helpfulness,clarity,easiness,quality
        for label in ['profgender','profhotness','online',\
                      'interest','textbookuse']:
            index = abs(hash(label+'_'+row[label])) % D
            x.append(index)
     
        # remove duplicated indexes
        x = list(set(x))

        yield int(ID), x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = gradient_descent(alpha, beta, L1, L2, D, interaction)

# start training
print('Training Learning started')
for e in range(epoch):
    loss = 0.
    count = 0
    for t,  x, y in data(train, D):  # data is a generator
        p = learner.predict(x) # predincting data
        loss += logloss(p, y) # measuting the loss
        learner.update(x, p, y) # updating the model
        count+=1
        if count%15000==0:
            print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), count, loss/count))

##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
count=0
print('Testing started')
with open(submission, 'w') as outfile:
    outfile.write('id,quality\n')
    for  ID, x, y in data(test, D):
        count+=1
        if count%15000==0:
            print('%s\tencountered: %d' % (datetime.now(), count))
        # we should re-normalized the prediction to be between 2 and 10
        p = 8.*learner.predict(x) + 2.0
        outfile.write('%s,%s\n' % (ID, str(p)))
#rmse.calc_rmse('sgd_subm1.csv')

