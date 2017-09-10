
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# fix random seed for reproducibility
np.random.seed(17)
sns.set(style='white', context='notebook', palette='deep')

from itertools import chain 
from sklearn import preprocessing
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, learning_curve

# load data
data = pickle.load(open("esSequencesWin2.pkl", 'rb'))

# load word2vec model
# THE FOLLOWING PATH SHOULD BE PROVIDED AND UN_COMMENTED
# model = Word2Vec.load(THE PATH TO WORD2VEC)

# fill this in cases where a word is outside the vocabulary 
nul = np.array([0] * 300)

X_ = [s[0] for s in data]
X = []
for seq in X_:
    temp = []
    for i in range(2, len(seq)):
        try:
            temp.append(model[seq[i]])
        except KeyError:
            # insert the nul vector 
            temp.append(nul) 
    X.append(temp)
# merge all the word2vec vectors in each sequence (length of each should be 300 * 4 = 1200)
X = np.array([list(chain.from_iterable(lst)) for lst in X])

# standardize the data attributes
standardized_X = preprocessing.scale(X)

y = np.array([s[2] for s in data])


random_state = 2
classifiers = []

classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(SVC(random_state=random_state))

clfName = ["LR","LDA","DT", "RF", "MLP","SVC"]

cv_results = []
from typeawareCV import *
cv_results = typeaware_cross_val_multiClassifier(classifiers, X, y, data)

cv_means = []
cv_std = []
for i in range(len(cv_results)):
    with open('performance_compare.txt', 'a') as f:
        f.write(str(clfName[i])+"\n")
        f.write(str(cv_results[i]))
        f.write("\n======\n")
        f.write("accuracy: ")
        f.write(str(cv_results[i].mean()))
        f.write("\n")
        f.write("std: ")
        f.write(str(cv_results[i].std()))
        f.write("\n======\n")
    cv_means.append(cv_results[i].mean())
    cv_std.append(cv_results[i].std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LR","DT",
"RF", "MLP","SVC"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Accuracy")

cv_res.to_csv('clf_typeAware_res_es.csv')

plt.show()
