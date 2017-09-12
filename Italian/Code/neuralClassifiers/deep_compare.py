"""Using this script we compared performances of several different neural network classifiers
    The best performing model is the ConvNet+LSTM whose result gets printed here. Uncomment to see
    a full comparison of classifiers"""

# coding: utf-8

# In[1]:

import pickle
import numpy as np

# fix random seed for reproducibility
np.random.seed(17)

from itertools import chain  
from gensim.models import Word2Vec
from preprocess import *
from keras.optimizers import SGD
from keras import regularizers
from keras.preprocessing.text import one_hot,text_to_word_sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# load data
data = pickle.load(open("../../Data/sequencesWin2.pkl", 'rb'))

# load word2vec model
# PROVIDE THE PATH AND COMMENT OUT THE FOLLOWING LINE
# model = Word2Vec.load(THE PATH TO WORD2VEC)

X = []

for seq, lm, lb in data:
    X.append(seq)

X = np.array(X)

# list of all unique words (len 22292)
word_list = list(set(chain(*X)))

# indexing all the words 
words_indices = dict((w, i) for i, w in enumerate(word_list))
indices_words = dict((i, w) for i, w in enumerate(word_list))

embedding_dimension = 300
embedding_matrix = np.zeros((len(words_indices), embedding_dimension)) # +1 removed 

j = 0
for word, i in words_indices.items():
    try:
        embedding_vector = model.wv[word]
    except KeyError:
        embedding_vector = None 
        
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]

sentence_len = 6
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=sentence_len)

# replacing words in each sequence with its corresponding unique index 
def word_to_array(word_list):
    return list(map(lambda word: words_indices[word], word_list))


# In[9]:

# single-layer LSTM with 100 neurons and drop-out = 0.1
def single_lstm():
    lstm_model = Sequential()
    lstm_model.add(embedding_layer)
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=False)))
    lstm_model.add(Dropout(0.7))
    lstm_model.add(Dense(1))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return lstm_model


# In[ ]:

# two-layered LSTM with 100 neurons and drop-out = 0.1
def double_lstm():
    lstm_model = Sequential()
    lstm_model.add(embedding_layer)
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=True)))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=False)))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(1))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return lstm_model


# In[ ]:

def single_convnet():
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", 
                        filters=100, 
                        kernel_size=3, 
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return cnn_model


# In[ ]:

def double_convnet():
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", 
                         filters=200, 
                         kernel_size=3, 
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(padding="same", 
                         filters=100, 
                         kernel_size=3, 
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return cnn_model


# In[ ]:

def convnet_lstm():
    mixed_model = Sequential()
    mixed_model.add(embedding_layer)
    mixed_model.add(Conv1D(padding="same", 
    	                   filters=1000, 
    	                   kernel_size=3, 
    	                   activation="relu",
    	                   # kernel_regularizer=regularizers.l2(0.01),
                           # activity_regularizer=regularizers.l1(0.01)
                           ))
    mixed_model.add(MaxPooling1D(pool_size=6))
    mixed_model.add(Bidirectional(LSTM(500, return_sequences=False)))
    mixed_model.add(Dropout(0.5))
    mixed_model.add(Dense(1))
    mixed_model.add(Activation('sigmoid'))
    mixed_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return mixed_model


# In[ ]:

models = [single_lstm, double_lstm, single_convnet, double_convnet, convnet_lstm]


# In[3]:

# performs type-aware startified cross validation on a neural network model
# returns mean and std of the accuracy 

def cross_validator(model, data): 
    train_datas, test_datas  = train_test(data)
    nn_model = model()
    nn_model.save_weights('init_weights.h5')
    evals = []
    for i in range(10):
        nn_model = model()
        train_data, test_data  = train_datas[i], test_datas[i]
        X_train = []
        y_train = []
        for seq, lemma, label in train_data:
            X_train.append(word_to_array(seq))
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = []
        y_test = []
        for seq, lemma, label in test_data:
            X_test.append(word_to_array(seq))
            y_test.append(label)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        nn_model.fit(X_train,
                     y_train,
                     batch_size=int(len(X_train)/2),
                     epochs=300,
                     validation_split=0.1,
                     verbose=1)
        evali = nn_model.evaluate(X_test,y_test,batch_size=int(len(X_train)/2))
        evals.append(evali[1])
        # reset the weights so the next model will modify the weights from scratch
        nn_model.load_weights('init_weights.h5')
    return np.mean(evals), np.std(evals)


# In[ ]:

results = []

# import pandas as pd 

# import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

# import seaborn as sns
# sns.set(style='white', context='notebook', palette='deep')

# for model in models:
#     results.append(cross_validator(model, data))
# results = list(zip(*results))

# cv_means = list(results[0])
# cv_std = list(results[1])

# cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["1-layer LSTM","2-layer LSTM","1-layer ConvNet",
# "2-layer ConvNet", "ConvNet LSTM"]})

# g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
# g.set_xlabel("Mean Accuracy")
# g = g.set_title("Cross validation scores")


# # In[ ]:

# plt.show()


# # In[10]:

result = cross_validator(convnet_lstm, data)
print('result=', result)

# # In[ ]:

# results.append(result)


# # In[5]:

# result


# # In[6]:

# results


# # In[ ]:



