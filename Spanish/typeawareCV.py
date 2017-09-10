import pickle
import random
import numpy as np

# the typeDist file contains each expression's type, its frequency and the percentage of time 
# the expression was annotated by '1' (as opposed to '0')
def readTypes():
	types = {}
	with open("typeDist_es.txt", "r", encoding="utf8") as f:
		lines = f.readlines()
		for l in lines:
			l1 = l.split('\t')
			# this weeds out expressions with a frequency of lower than 3 
			if int(l1[1])>0:
				types[l1[0]] = float(l1[2])
	return types
	 
# seqs is a list of tuples of this form: 
# (expression occurrence with its neighboring words, expression type, MWE label)



def typeaware_cross_val(classifier, X, y, seqs): # our ten folds for cross-validation
        
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

def typeaware_cross_val_multiClassifier(classifiers, X, y, seqs): # our ten folds for cross-validation
                                                                # in the same format of sklearn cross-val
                                                                # it takes several classifiers as input and
                                                                # gives scores for all of them
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
                print(np.array(scores).mean())
        return allClfScores
	
