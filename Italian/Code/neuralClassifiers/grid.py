"""tests a range of hyper-parameters with different neural network architectures"""

# coding: utf-8

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
from keras.wrappers.scikit_learn import KerasClassifier

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

# single-layer LSTM 
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

# two-layered LSTM 
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

# single-layer ConvNet
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

# double ConvNet
def double_convnet():
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", 
                         filters=100, 
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

# ConvNet+LSTM hybrid 
def convnet_lstm():
    mixed_model = Sequential()
    mixed_model.add(embedding_layer)
    mixed_model.add(Conv1D(padding="same", 
    	                   filters=500, 
    	                   kernel_size=3, 
    	                   activation="relu",
    	                   kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01)))
    mixed_model.add(MaxPooling1D(pool_size=2))
    mixed_model.add(Bidirectional(LSTM(300, return_sequences=False)))
    mixed_model.add(Dropout(0.6))
    mixed_model.add(Dense(1))
    mixed_model.add(Activation('sigmoid'))
    mixed_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return mixed_model


# In[ ]:

models = [single_lstm, double_lstm, single_convnet, double_convnet, convnet_lstm]


# In[3]:

# performs type-aware cross validation on a neural network model
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
                     batch_size=len(X_train),
                     epochs=300,
                     validation_split=0.1,
                     verbose=1)
        evali = nn_model.evaluate(X_test,y_test,batch_size=len(X_train))
        evals.append(evali[1])
        nn_model.load_weights('init_weights.h5')
    return np.mean(evals), np.std(evals)


# In[ ]:

results = []

result = cross_validator(convnet_lstm, data)
print('result=', result)



# model whose hyperparameters are to be optimized 
model_opt = KerasClassifier(build_fn=convnet_lstm, epochs=50)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model_opt, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)
