'''
This part of code compares the performances of different traditional classifiers:
Logistic Regression, Decision Tree, Random Forest, Multiple Layer Perceptron
and SVM.

The performances are computed using regular cross-validation
'''
import pickle
import numpy as np
import pandas as pd
import random

# fix random seed for reproducibility
np.random.seed(17)

from itertools import chain 
from sklearn import preprocessing
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

# load data
data = pickle.load(open("../../Data/sequencesWin2.pkl", 'rb'))
random.shuffle(data)

"""load word2vec model"""
# THE FOLLOWING PATH SHOULD BE PROVIDED AND COMMENTED OUT 
# model = Word2Vec.load(PATH TO WORD2VEC)

# fill this in cases where a word is outside the vocabulary 
nul = np.array([0] * 300)

import typeawareCV
selectedTypes = typeawareCV.readTypes()
X_ = [s[0] for s in data if s[1] in selectedTypes]
X = []
for seq in X_:
    temp = []
    for i in range(2, len(seq)):
        try:
            temp.append(model.wv[seq[i]])
        except KeyError:
            # insert the nul vector 
            temp.append(nul) 
    X.append(temp)
# merge all the word2vec vectors in each sequence (length of each should be 300 * 4 = 1200)
X = np.array([list(chain.from_iterable(lst)) for lst in X])

# standardize the data attributes
standardized_X = preprocessing.scale(X)

y = np.array([s[2] for s in data if s[1] in selectedTypes])

# comparison of different classifiers   
random_state = 2
classifiers = []
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(SVC(random_state=random_state))

cv_results = []
cv_means = []
cv_std = []

# cross validate model with Kfold stratified cross val
kfold = KFold(n_splits=10)

for classifier in classifiers :
    sc = cross_val_score(classifier, standardized_X, y, scoring = "accuracy", cv = kfold)
    cv_results.append(sc)
    cv_means.append(sc.mean())
    cv_std.append(sc.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LR","DT",
"RF", "MLP","SVC"]})

cv_res.to_csv('clf_regular_res')
