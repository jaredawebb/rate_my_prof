# https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
# modified by rcarson available as FTRL starter on kaggle code 
# https://www.kaggle.com/jiweiliu/springleaf-marketing-response/ftrl-starter-code
##############################################################################
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import numpy as np
import pandas as pd
import pickle

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


def data(path, D, stopwords):
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
            del row['tid']
        except:
            pass

        # normalized quality \in [0 - 1]
        try: 
            y = (float(row['quality']) - 2.0) / 8.0
            del row['quality']
            del row['easiness']
            del row['clarity']            
        except: 
           y = 0    

        # build x
        x = []
        
        #labels: 
        #id,tid,dept,date,forcredit,attendance,textbookuse,interest,
        #grade,tags,comments,helpcount,nothelpcount,online,profgender,
        #profhotness,helpfulness,clarity,easiness,quality
        key = 'comments'
        if row[key] != "No Comments":
            value = [w for w in row[key].lower().split() if w not in stopwords]
            for word in value:
                x.append(abs(hash(word)) % D)
        else:
            value = -1
            x.append(value)
        del row['comments']
        for key in row:#['profhotness', 'interest', 'online']:
            value = row[key]
            index = abs(hash(key + '_' + str(value))) % D
            x.append(index)
        # remove duplicated indexes
        x = list(set(x))

        yield int(ID), x, y


##############################################################################
# start training #############################################################
##############################################################################
'''
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
    probs = []
    for  ID, x, y in data(test, D):
        count+=1
        if count%15000==0:
            print('%s\tencountered: %d' % (datetime.now(), count))
        # we should re-normalized the prediction to be between 2 and 10
        probs.append(learner.predict(x))
    
    max_p = max(probs)

    count = 0
    for ID, x, y in data(test, D):
        outfile.write('%s,%s\n' % (ID, str(8*probs[count] + 2)))
        count += 1
'''
