
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

def word2features(seq, i):

        features = {}
        features['word'] = seq[i][0] 
        
        if i==2:
                    features['POS'] = "V"
                    features['verb'] = seq[i][0]
                    features['noun'] = seq[i+1][0]
                    features['w+'] = seq[i+2][0]
                    features['w++'] = seq[i+3][0]
        '''
        if i>0:
                    features['-1:word'] = seq[i-1][0]
        else:
                    features['BOS'] = True
        if i < len(seq)-1:
                    features['+1:word'] = seq[i+1][0]
        else:
                    features['EOS'] = True
        '''                                                         
        return features

def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
        return [label for word, label in sent]

def sent2tokens(sent):
        return [token for word, label in sent]


def bio_classification_report(y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
                
        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
                y_true_combined,
                y_pred_combined,
                labels = [class_indices[cls] for cls in tagset],
                target_names = tagset,
        )

################################################
################################################
################################################

def skCRFclassify(sf):
          os = 0
          seqs = []
          with open(sf, 'rb') as f:
                  seqs = pickle.load(f)
          ### Reading the whole data in the format of crfsuite
          ### list of sentences, each sentence containing tuples: (word, label)
          labeledSeqs = []
          for s in seqs:
                  ls = []
                  if s[2]==0:
                          for w in s[0]:
                              ls.append((w, 'O'))
                              os+=1
                  elif s[2]==1:
                          for w in s[0][0:2]:
                                  ls.append((w,'O'))
                                  os+=1
                          ls.append((s[0][2], 'B'))
                          ls.append((s[0][3], 'I'))
                          for w in s[0][4:]:
                              ls.append((w,'O'))
                              os+=1
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
          scores = typeaware_cross_val_CRF(crf, X, y, seqs)

          ### USE THE FOLLOWING LINE FOR REGULAR CROSS VALIDATION
          #scores = rand_cross_val_CRF(crf, X, y, 10)

          ### USE THE FOLLOWING LINE FOR SKLEARN CROSS VALIDATION
          #scores = cross_val_score(crf, X, y, n_jobs=-1, cv=10)
          
          print(scores)
          print(np.mean(scores), np.std(scores))
          
skCRFclassify('sequencesFinalWin2.pkl')
 
