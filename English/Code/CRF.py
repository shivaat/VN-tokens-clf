""" This file contains functions for the CRF classfier"""

# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from collections import Counter
import gc
import codecs

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import cross_val_score
import sklearn
import sklearn_crfsuite

from typeawareCV import *

""" This function is used to extract Verb + Noun (s) from BNC sequences that are tagged
    in VNC-Tokens dataset """
def getVN(sent,vn):
          verb = vn.split('_')[0]
          noun = vn.split('_')[1]
          v = -1
          n = -1
          for i in range (len(sent)):
                    if sent[i][1] == verb and v == -1:
                              v = i
                    elif sent[i][1] == verb and v != -1 and n==-1:
                    ### if v has got a value before, it can get a new value
                    ### only if n has not yet got a value
                              v = i                            
                    elif sent[i][1] == noun and v != -1 and n==-1:
                              n = i
          return v, n

"""For each word in the sequence, this function selects features to be fed into CRF"""
def word2features(seq, i):
        features = {}
        features['word'] = seq[i][0][0] 

#        if i==2:
#                    features['POS'] = "V"
#                    features['verb'] = seq[i][0]
#                    features['noun'] = seq[i+1][0]
#                    features['w+'] = seq[i+2][0]
#                    features['w++'] = seq[i+3][0]
        #if i==3:
        #            features['POS'] = "N"
        
        if i>0:
                    #features['-1:word'] = seq[i-1][0][0]
                    a=1
        else:
                    features['BOS'] = True
        if i < len(seq)-1:
                    features['+1:word'] = seq[i+1][0][0]
        else:
                    features['EOS'] = True
        if i<len(seq)-2:
                    features['+2:word'] = seq[i+2][0][0] 
        return features

def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
        return [label for word, label in sent]

def sent2tokens(sent):
        return [token for word, label in sent]


#def train_test(X_train, y_train, X_test, y_test):
def train_test(X_train, y_train, testData, X_test):
        crf = sklearn_crfsuite.CRF(
        algorithm='arow', 
        #c1=0.1, # coefficient for L1 penalty
        #c2=1e-3, # coefficient for L2 penalty
        max_iterations=50,
         # include transitions that are possible, but not observed 
        all_possible_transitions=True)
        crf.fit(X_train, y_train)
        y_pred = crf.predict(X_test)

        return y_pred

################################################
################################################
################################################


### Reading the whole data in the format of crfsuite
### list of sentences, each sentence containing tuples: (word, label)
os = 0
seqs = []
with open("../Data/taggedSentsBNC.pkl", 'rb') as f:
          seqs = pickle.load(f)
newSeqs = []
labeledSeqs = []
for s in seqs:
          ls = []
          if s[2]=='L':
                  for w in s[0]:
                      ls.append((w, 'O'))
                      os+=1
                  labeledSeqs.append(ls)
          elif s[2]=='I':
                  v, n = getVN(s[0], s[1])
                  if v!=-1 and n!=-1:
                          for w in s[0][0:v]:
                                  ls.append((w,'O'))
                                  os+=1
                          ls.append((s[0][v], 'B'))
                          for w in s[0][v+1:n]:
                                  ls.append((w,'O'))
                                  os+=1
                          ls.append((s[0][n], 'I'))
                          for w in s[0][n+1:]:
                              ls.append((w,'O'))
                              os+=1
                  newSeqs.append(s)
                  labeledSeqs.append(ls)

X = [sent2features(s) for s in labeledSeqs]
y = [sent2labels(s) for s in labeledSeqs]

crf = sklearn_crfsuite.CRF(
  algorithm='arow', 
  #c1=0.1, # coefficient for L1 penalty
  #c2=1e-3, # coefficient for L2 penalty
  max_iterations=50,
   # include transitions that are possible, but not observed 
  all_possible_transitions=True)


### USE THE FOLLOWING LINE FOR TYPE-AWARE CROSS VALIDATION
scores = typeaware_cross_val_CRF(crf, X, y, newSeqs)

### USE THE FOLLOWING LINE FOR REGULAR CROSS VALIDATION
#scores = rand_cross_val_CRF(crf, X, y, 10)

### USE THE FOLLOWING LINE FOR BUILT-IN SKLEARN CROSS VALIDATION
#scores = cross_val_score(crf, X, y, n_jobs=-1, cv=10)

print(scores)
print(np.mean(scores), np.std(scores))
