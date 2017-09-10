'''
This part of code compare the performance of different traditional classifiers:
Logistic Regression, Decision Tree, Random Forest, Multiple Layer Perceptron
and SVM.
The performance is computed using regular cross-validation
'''
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# fix random seed for reproducibility
np.random.seed(17)
sns.set(style='white', context='notebook', palette='deep')
sns.set(rc={"figure.figsize": (3, 3)})

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

# load data
data = pickle.load(open("esSequencesWin2.pkl", 'rb'))
random.shuffle(data)

# load word2vec model
# THE FOLLOWING PATH SHOUL BE PROVIDE AND UN_COMMENTED
# model = Word2Vec.load(THE PATH TO WORD2VEC)

# fill this in cases where a word is outside the vocabulary 
nul = np.array([0] * 300)

import typeawareCV
selectedTypes = ourStratifiedCV.readTypes()
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

# Cross validate model with Kfold stratified cross val
kfold = KFold(n_splits=10)

# Modeling step Test differents algorithms 
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
for classifier in classifiers :
    sc = cross_val_score(classifier, standardized_X, y, scoring = "accuracy", cv = kfold)
    cv_results.append(sc)
    cv_means.append(sc.mean())
    cv_std.append(sc.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LR","DT",
"RF", "MLP","SVC"]})

cv_res.to_csv('clf_regular_res_es')

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Accuracy")
# g = g.set_title("(a)")

plt.show()
