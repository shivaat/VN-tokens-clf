""" This file includes functions that perfom different forms of cross-validation.
	Each cross-validation reads expression types and accordingly divides instances into
	separate folds """

import pickle
import random
import numpy as np

# the typeDist file contains each expression's type, its frequency and the percentage of time 
# the expression was annotated by '1' (as opposed to '0')
def readTypes():
	types = {}
	with open("../../Data/typeDist_es.txt", "r", encoding="utf8") as f:
		lines = f.readlines()
		for l in lines:
			l1 = l.split('\t')
			# this weeds out expressions with a frequency of lower than 3 
			if int(l1[1])>0:
				types[l1[0]] = float(l1[2])
	return types
	 
# seqs is a list of tuples of this form: 
# (expression occurrence with its neighboring words, expression type, MWE label)

"""custom (type-aware) 10-fold cross-validation in the same format as sklearn's cross-val"""
def typeaware_cross_val(classifier, X, y, seqs): 
        scores = []
        types = list(readTypes().keys())
        random.shuffle(types)
        testTypeSize = int(len(types)/10)
        for i in range(10):
                X_train, X_test, y_train, y_test = [], [], [], []
                for j in range(len(seqs)):
                        if seqs[j][1] in types[testTypeSize*i:testTypeSize*(i+1)]:
                                X_test.append(X[j])
                                y_test.append(y[j])
                        elif seqs[j][1] in types:
                                X_train.append(X[j])
                                y_train.append(y[j])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_test = np.array(X_test)
                y_test = np.array(y_test)

                classifier.fit(X_train, y_train)
                sc = classifier.score(X_test, y_test)
                scores.append(sc)
        return np.array(scores)

"""similar to typeaware_cross_val, but receives a list of classifiers and returns the results for all of them"""
def typeaware_cross_val_multiClassifier(classifiers, X, y, seqs): 
        scores = []
        types = list(readTypes().keys())
        random.shuffle(types)
        testTypeSize = int(len(types)/10)
        folds = []
        for i in range(10):
                X_train, X_test, y_train, y_test = [], [], [], []
                for j in range(len(seqs)):
                        if seqs[j][1] in types[testTypeSize*i:testTypeSize*(i+1)]:
                                X_test.append(X[j])
                                y_test.append(y[j])
                        elif seqs[j][1] in types:
                                X_train.append(X[j])
                                y_train.append(y[j])

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                folds.append((X_train, y_train, X_test, y_test))

        allClfScores = []
        for classifier in classifiers:
                scores = []
                for i in range(10):
                        classifier.fit(folds[i][0], folds[i][1])
                        sc = classifier.score(folds[i][2], folds[i][3])

                        scores.append(sc)
                allClfScores.append(np.array(scores))
        return allClfScores
	
"""this evaluation function is used only for the CRF classifier"""
def evaluate(pred, yTest):
        acc = 0
        for i in range(len(yTest)):
                if yTest[i][2] == pred[i][2]: # This checks only the beginning of VN expression, which starts from index 2 of a sequence
                        acc+=1
        return acc/len(yTest)

"""type-aware cross-validation for the CRF classifier"""
def typeaware_cross_val_CRF(classifier, X, y, seqs): 
        scores = []
        types = list(readTypes().keys())
        random.shuffle(types)
        testTypeSize = int(len(types)/10)
        for i in range(10):
                X_train, X_test, y_train, y_test = [], [], [], []
                for j in range(len(seqs)):
                        if seqs[j][1] in types[testTypeSize*i:testTypeSize*(i+1)]:
                                X_test.append(X[j])
                                y_test.append(y[j])
                        else:
                                X_train.append(X[j])
                                y_train.append(y[j])

                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                X_test = np.array(X_test)
                y_test = np.array(y_test)

                classifier.fit(X_train, y_train)

                y_pred = classifier.predict(X_test)
                sc = evaluate(y_pred, y_test)

                scores.append(sc)
        return np.array(scores)

"""standard random cross-validation"""
def rand_cross_val_CRF(classifier, X, y, cv):
        seqs = list(zip(X, y))
        fSize = int(len(seqs)/cv)
        np.random.shuffle(seqs)
        scores = []
        for i in range(cv):
                test = seqs[fSize*i:fSize*(i+1)]
                train = seqs[0:fSize*i]+seqs[fSize*(i+1):]
                test1 = list(zip(*test))
                train1 = list(zip(*train))

                X_test = np.array(test1[0])
                y_test = np.array(test1[1])
                X_train = np.array(train1[0])
                y_train = np.array(train1[1])

                classifier.fit(X_train, y_train)
                #sc = classifier.score(X_test, y_test)

                y_pred = classifier.predict(X_test)
                sc = evaluate(y_pred, y_test)

                scores.append(sc)
        return np.array(scores)
