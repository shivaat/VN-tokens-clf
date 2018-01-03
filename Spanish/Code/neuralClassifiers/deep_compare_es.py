"""Using this script we compared performances of several different neural network classifiers.
    The best performing model is the ConvNet+LSTM
    The results can be reported both using regular and type-aware cross-validation.
    Relevant lines should be commented/uncommented"""

# coding: utf-8

import pickle
import numpy as np

# fix random seed for reproducibility
np.random.seed(17)

from itertools import chain  
from gensim.models import Word2Vec
from train_test_splitting import *
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

# load data
data = pickle.load(open("../../Data/esSequencesWin2_agreed.pkl", 'rb'))
random.shuffle(data)

# load word2vec model
# PROVIDE THE PATH AND COMMENT OUT THE FOLLOWING LINE
# model = Word2Vec.load(THE PATH TO WORD2VEC)

X = []
for seq, lm, lb in data:
    X.append(seq)

X = np.array(X)

######-----Add a word embedding layer at the top, based on pretrained weights-----######

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

######--------------------------------------------------------------------------######

# replacing words in each sequence with its corresponding unique index 
def word_to_array(word_list):
    return list(map(lambda word: words_indices[word], word_list))

def feed_forward():
    ff_model = Sequential()
    ff_model.add(embedding_layer)
    ff_model.add(Dense(300, kernel_initializer='normal', activation='relu', input_dim=6))
    ff_model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    ff_model.add(Dropout(0.7))
    ff_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    ff_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return ff_model

def single_lstm():
    lstm_model = Sequential()
    lstm_model.add(embedding_layer)
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=True, dropout=0.3)))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(Flatten())
    lstm_model.add(Dense(1))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return lstm_model

def double_lstm():
    lstm_model = Sequential()
    lstm_model.add(embedding_layer)
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=True)))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(Bidirectional(LSTM(300, return_sequences=False)))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(Dense(1))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return lstm_model

def single_convnet():
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", 
                        filters=300, 
                        kernel_size=4, 
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=4))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return cnn_model

def double_convnet():
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", 
                        filters=300, 
                        kernel_size=4, 
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=3))
    cnn_model.add(Conv1D(padding="same", 
                        filters=300, 
                        kernel_size=4, 
                        activation="relu",
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return cnn_model

def convnet_lstm():
    mixed_model = Sequential()
    mixed_model.add(embedding_layer)
    mixed_model.add(Conv1D(padding="same", filters=300, kernel_size=4, activation="relu"))
    mixed_model.add(MaxPooling1D(pool_size=4))
    mixed_model.add(Bidirectional(LSTM(300, return_sequences=False)))
    mixed_model.add(Dropout(0.6))
    mixed_model.add(Dense(1))
    mixed_model.add(Activation('sigmoid'))
    mixed_model.compile(loss='binary_crossentropy',optimizer='adam', metrics=["accuracy"])
    return mixed_model


""" This fuction receives the neural model and the data and performs regular or type-aware cross-validation
    to report the results"""
def cross_validator(model, data):
    ### The following line for type-aware fold splitting
    train_datas, test_datas  = train_test(data)

    ### The following 2 lines should be commented out for random fold splitting
    train_datas, test_datas  = random_kfold(data)
    
    nn_model = model()
    nn_model.save_weights('init_weights_es.h5')
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
                     batch_size=5000,
                     epochs=100,
                     validation_split=0.1,
                     verbose=1)
        evali = nn_model.evaluate(X_test,y_test,batch_size=5000)
        evals.append(evali[1])
        nn_model.load_weights('init_weights_es.h5')
    return np.mean(evals), np.std(evals)

a = cross_validator(convnet_lstm, data)

with open('res.txt', 'w') as f:
    f.write(str(a))



