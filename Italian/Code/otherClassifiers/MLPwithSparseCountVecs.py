"""This code is for performing the classification using Multiple Layer Perceptron using count based vectors as input"""
# coding: utf-8

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Load data
data = pickle.load(open("../../Data/sequencesFinalWin2.pkl", 'rb'))

#  s[0][2:] to match our word2vec vectors of the verb, the noun and the 2 words after
X_ = [s[0][2:] for s in data]
X = [" ".join(x) for x in X_]

y = np.array([s[2] for s in data])

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
X = vectorizer.fit_transform(X).toarray()

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
X = tfidf.toarray() 

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10)

random_state = 17
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(random_state=random_state)

### The following line for type-aware fold splitting
from typeawareCV import *
result = typeaware_cross_val(classifier, X, y, data)

### The following 2 lines should be commented out for random fold splitting 
#result = cross_val_score(classifier, X, y, cv=kf)

print('mean accuracy', result.mean())
print('mean STD:', result.std())

