'''
This part of code compares the performances of different traditional classifiers:
Logistic Regression, Decision Tree, Random Forest, Multiple Layer Perceptron and SVM.
The performances are computed using regular cross-validation
'''
# coding: utf-8

import pickle, gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random, spacy

# fix random seed for reproducibility
np.random.seed(17)
sns.set(style='white', context='notebook', palette='deep')

from itertools import chain 
from sklearn import preprocessing
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, learning_curve

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
     
""" This function extracts the words of our interest from sequences of BNC that are tagged
    in VNC-Tokens dataset. The words of our interest is the Verb, the Noun and two words after"""
def getWords(sent):
          v, n = getVN(s[0], s[1])
          if v!=-1 and n!=-1:
                    verb = s[0][v][0].strip()
                    noun = s[0][n][0].strip()
                    if n+1 < len(s[0]):
                              w1 = s[0][n+1][0].strip()
                              if n+2 < len(s[0]):
                                        w2 = s[0][n+2][0].strip()
                              else:
                                        w2 = 'NULL'
                    else:
                              w1 = 'NULL'
                              w2 = 'NULL'
                    return verb,noun,w1,w2
          return 0

# load data
sents = pickle.load(open(".Data/taggedSentsBNC.pkl",'rb'))
seqs = []
# We don't consider the Q susbet in the dataset
for s in sents:
          if s[2] in ['L', 'I'] and getWords(s):
                    w1,w2,w3,w4 = getWords(s)
                    if s[2]=='L':
                         seq = ([w1,w2,w3,w4], s[1], 0)
                    elif s[2]=='I':
                         seq = ([w1,w2,w3,w4], s[1], 1)
                    seqs.append(seq)
random.shuffle(seqs)

""" Loading word vectors """
# For English we have used the English pre-trained vectors of spacy
# It loads English tokenizer, tagger, parser, NER and word vectors
glove = spacy.load('en')

X_ = [seq[0] for seq in seqs]

X = []
for x in X_:
     try:
          word_1 = glove(x[0]).vector 
     except KeyError:
          word_1 = np.array([0] * 300)
     try:
          word_2 = glove(x[1]).vector
     except KeyError:
          word_2 = np.array([0] * 300)
     try:
          word_3 = glove(x[2]).vector
     except KeyError:
          word_3 = np.array([0] * 300)
     try:
          word_4 = glove(x[3]).vector
     except KeyError:
          word_4 = np.array([0] * 300)
     X.append( np.concatenate((word_1, word_2, word_3, word_4), axis=0)  )

# construct y 
y = [seq[2] for seq in seqs]

'''
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y = lb.fit_transform(y)
y = np.array(y).ravel()
'''

random_state = 17
classifiers = []

classifiers.append(LogisticRegression(C=0.10000000000000001, solver='sag', penalty='l2', max_iter=1000, tol= 0.009))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(SVC(random_state=random_state))

cv_results = []

# Cross validate model with Kfold stratified cross val
kfold = KFold(n_splits=10)

from sklearn.model_selection import cross_val_score

for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X, y, scoring = "accuracy", cv = kfold))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"Means":cv_means,"STD": cv_std,"Algorithm":["LR","DT","RF", "MLP","SVC"]})

g = sns.barplot("Means","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Accuracy")

cv_res.to_csv('VNC_clf_regular_res')

plt.show()
