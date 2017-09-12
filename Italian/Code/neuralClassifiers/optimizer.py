"""Search for suitable hyper-parameters for the hybrid ConvNet+BiLSTM model"""

# coding: utf-8

from itertools import product
from keras.optimizers import RMSprop, Adadelta, Adamax, Nadam, TFOptimizer

dropouts = [0.5, 0.6, 0.7, 0.4]
kernel_sizes = [3, 4, 5, 6]
optimizers = ['adam', 'adagrad']
filters = [100, 300]
epochs = [100]
lstm_units = [100, 200, 300]
pool_sizes = [6,5,4,2] # maximum num of pool_size can't be bigger than sentence length (6)


s = [dropouts, kernel_sizes, optimizers, filters, epochs, lstm_units, pool_sizes]

# each element of params will be of the shape:
# (dropouts, kernel_sizes, optimizers, filters, epochs, lstm_units)
params = list(product(*s))

# print(len(params))

import pickle
import numpy as np

# fix random seed for reproducibility
np.random.seed(17)

from itertools import chain  
from gensim.models import Word2Vec
from preprocess import *
from keras.optimizers import SGD
from keras.preprocessing.text import one_hot,text_to_word_sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
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

# list of all unique words cnn_model
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


def build_model(param):
    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(padding="same", filters=param[3], kernel_size=param[1], activation="relu"))
    cnn_model.add(MaxPooling1D(pool_size=param[6]))
    cnn_model.add(Bidirectional(LSTM(param[5], return_sequences=True)))
    cnn_model.add(Flatten())
    cnn_model.add(Dropout(param[0]))
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))


    cnn_model.compile(loss='binary_crossentropy',optimizer=param[2], metrics=["accuracy"])

    # adam
    return cnn_model    

# try all the permutations of the parameters, run the models accordingly, save the results along 
# with the parameters in a dictionary 
results = dict()
# each param structure: 
# (dropouts, kernel_sizes, optimizers, filters, epochs, lstm_units)
for param in params:
	evals = []
	# we want to repeat the random selection of train and test data for 100 iterations
	# and finally get the overall average of the accuracy of the model while considering the 
	# overall standard deviation

	train_datas, test_datas  = train_test(data)

	cnn_model = build_model(param)
	cnn_model.save_weights('init_weights_optimizer.h5')

	for i in range(10):
	    cnn_model = build_model(param)
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
	    # early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
	    cnn_model.fit(X_train,y_train,
	              batch_size=len(X_train),epochs=param[4],
	              validation_split=0,
	              verbose=1,
	              # callbacks=[early_stop])
	              )
	    
	    evali = cnn_model.evaluate(X_test,y_test,batch_size=len(X_test))
	    evals.append(evali[1])
	    # reset the weights of the model 
	    cnn_model.load_weights('init_weights_optimizer.h5')
	result = np.mean(evals)
	results[param] = result
	print('results at this point:', results)
	with open('results.txt', 'w') as f:
		f.write(str(results))
	best = max(results.keys(), key=(lambda k: results[k]))
	with open('best_parameter.txt', 'w') as f:
		f.write(str(results[best]))
