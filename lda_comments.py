import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import pickle
import lda
import numpy as np

# Choose file to pull comments from
'''
datafile1 = './sample/sample_train.csv'
datafile2 = './sample/sample_test.csv'
sample = True
'''
datafile1 = './data/newtrain.csv'
datafile2 = './data/newtest.csv'
sample = False

# The stop words file gives a list of words that we
# do not want to consider in our analysis.
stop_words = './stop_words.txt'

# Read in stop words list
print('Reading Data')
with open(stop_words, 'r') as f:
	words = f.read()
	words = words.split('\n')[:-1]

# The count vectorizer object will give a document-word matrix of the bigrams
# present in the text.
count_vec = CountVectorizer(min_df=10, ngram_range=(1,2), stop_words=words)

# Load the datafiles with pandas
df_comm1 = pd.read_csv(datafile1, usecols=['id', 'comments'])
df_comm2 = pd.read_csv(datafile2, usecols=['id','comments'])

# It is important that we combine the comments from both files
# to learn the document-word matrix.  There might be words in the
# second file that are not in the first, which could cause
# problems later on.
df_comm = pd.concat([df_comm1, df_comm2])

# Learn the document-word matrix, call it cv_comm
print('Counting Words')
cv_comm = count_vec.fit_transform(df_comm['comments'].fillna(''))

# This is optional.  It doesn't take too long to do anyway, so
# I just leave it commented.
'''
# Save the matrix so we don't have to do it every single time.
pickle.dump(cv_comm, open(datafile[:-4] + '_dt_matrix.p', 'wb'))

# Save the ngrams as well just in case we want them.
picke.dump(count_vec.get_feature_names(), open(datafile[:-4] + '_feature_names.p', 'wb'))
'''

# Use LDA to find topic distributions in comments.
print('Running LDA')
n_topics=50
model = lda.LDA(n_topics=n_topics, n_iter=50, random_state = 1)
model = model.fit(cv_comm)

# print most likely words in each topic distributions.
'''
n_top_words = 8
vocab = np.array(count_vec.get_feature_names())
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
	topic_words = vocab[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))
'''
##########################################################
# Train a ridge regression with the topic distributions. #
##########################################################
'''
print('Training Ridge Regression')
Ytrain = pd.read_csv(datafile1, usecols=['quality'])  # We want to train against quality

cl = sklearn.linear_model.Ridge()
# These are the topic disributions for each comment.
'''
doc_topic = model.doc_topic_
'''
# Fit on the training set.
f_data = pd.read_csv('./data/newtrain.csv')
f_data.drop(['id', 'tid', 'helpfulness', 'clarity', 'easiness','comments', 'quality'], axis=1, inplace=True)
f_data.replace(to_replace='nan', value=0, inplace=True)

final_data = np.hstack((np.array(doc_topic[:len(df_comm1)]), np.array(f_data))) 

cl.fit(final_data, np.array(Ytrain))

# Now predict on the test set.
print('Predicting')
f_data = pd.read_csv('./data/newtest.csv')
f_data.drop(['comments', 'id', 'tid'], axis=1, inplace=True)
f_data.replace(to_replace='nan', value=0, inplace=True)

final_data = np.hstack((np.array(doc_topic[len(df_comm1):]), np.array(f_data)))

Yhat = cl.predict(final_data)

# This data frame matches the topic numbers to their
# learned coefficients in the ridge regression.

df = pd.DataFrame(data={'topics':range(n_topics),
                        'coef':cl.coef_.flatten()[:n_topics]
                    })  
df.sort('coef',ascending=False,inplace=True)
nums_array = np.array(df.topics)

print("Ridge Coefficients")
print("Most positive:")

# Print the most helpful and most negative topic distributions.
for i, topic_dist in enumerate(topic_word[df.topics[0:5]]):
	topic_words = vocab[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(nums_array[i], ' '.join(topic_words)))

print("Most negative")

for i, topic_dist in enumerate(topic_word[df.topics[-5:]]):
	topic_words = vocab[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(nums_array[-5+i], ' '.join(topic_words)))


# Save results in kaggle format
submit = pd.DataFrame(data={'id': df_comm2.id, 'quality': np.ravel(Yhat)})
submit.to_csv('./submissions/submit_lda_ridge.csv', index = False)
'''
# Save restults for later calculations.
if sample:

	# Create dictionary
	topic_dict = {'id': df_comm1.id}
	for i in range(doc_topic.shape[1]):
		topic_dict['topic'+str(i+1)] = doc_topic[:len(df_comm1),i]
	submit = pd.DataFrame(data=topic_dict)
	submit.to_csv('./sample/train_topic_dist_n_' + str(n_topics) + '.csv', index = False)

	topic_dict = {'id':df_comm2.id}
	for i in range(doc_topic.shape[1]):
		topic_dict['topic'+str(i+1)] = doc_topic[len(df_comm1):,i]
	submit = pd.DataFrame(data=topic_dict)
	submit.to_csv('./sample/test_topic_dist_n_' + str(n_topics) + '.csv', index = False)

else:

	# Create dictionary
	topic_dict = {'id': df_comm1.id}
	for i in range(doc_topic.shape[1]):
		topic_dict['topic'+str(i+1)] = doc_topic[:len(df_comm1),i]
	submit = pd.DataFrame(data=topic_dict)
	submit.to_csv('./data/train_topic_dist_n_' + str(n_topics) + '.csv', index = False)

	topic_dict = {'id':df_comm2.id}
	for i in range(doc_topic.shape[1]):
		topic_dict['topic'+str(i+1)] = doc_topic[len(df_comm1):,i]
	submit = pd.DataFrame(data=topic_dict)
	submit.to_csv('./data/test_topic_dist_n_' + str(n_topics) + '.csv', index = False)



