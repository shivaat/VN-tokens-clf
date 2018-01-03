""" This file contains code for Naive Bayes classfier in both regular and type-aware scenario
"""
import pickle

""" load Data """ 
seqs = pickle.load(open("../../Data/esSequencesWin2_agreed.pkl",'rb'))

import nltk
import numpy as np
from sklearn import cross_validation

train_data = []
""" Extracting the Verb, the Noun and the two words after as features """
for seq in seqs:          
          train_data.append(({"verb":seq[0][0], "noun":seq[0][1], "w+":seq[0][2], "w++":seq[0][3]},int(seq[2])))

""" Computing the results using regular cross-validation """                        
cv = cross_validation.KFold(len(train_data), n_folds=10, shuffle=False, random_state=None)

scores = []
for traincv, testcv in cv:
          classifier = nltk.NaiveBayesClassifier.train(train_data[traincv[0]:traincv[len(traincv)-1]])
          scores.append(nltk.classify.util.accuracy(classifier, train_data[testcv[0]:testcv[len(testcv)-1]]))

""" Computing the results using type-aware cross-validation """
from typeawareCV import *
tpCVscores = typeaware_cross_val_nltk(nltk.NaiveBayesClassifier, train_data, seqs)
print("NB type_aware: ",np.mean(tpCVscores),np.std(tpCVscores))
