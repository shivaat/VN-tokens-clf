""" This file includes functions that perfom different forms of cross-validation.
	Each cross-validation reads expression types and accordingly divides instances
	into separate folds """

import pickle
import random
import numpy as np
import nltk

""" The number of expression types in VNC-Tokens are limited and are divide into three
    categories: Dev, Test an Skewed. We simply list them here, and then read them all
"""
def readTypes():
        types = {}
        Dev = ["blow_trumpet", "find_foot", "get_nod", "hit_road", "hit_roof", "kick_heel", "lose_head", "make_face", "make_pile", "pull_leg", "pull_plug", "pull_weight", "see_star", "take_heart"]
        Test = ["blow_top", "blow_whistle", "cut_figure", "get_sack", "get_wind", "have_word", "hit_wall", "hold_fire", "lose_thread", "make_hay", "make_hit", "make_mark", "make_scene", "pull_punch"]
        Skewed = ["blow_smoke", "bring_luck", "catch_attention", "catch_death",
                  "catch_imagination", "get_drift", "give_notice", "give_sack",
                  "have_fling", "have_future", "have_misfortune", "hold_fort",
                  "hold_horse", "hold_sway", "keep_tab", "kick_habit", "lay_waste",
                  "lose_cool", "lose_heart", "lose_temper", "make_fortune", "move_goalpost",
                  "set_fire", "take_root", "touch_nerve"]
        for item in Dev:
                types[item] = 'D'
        for item in Test:
                types[item] = 'T'
        for item in Skewed:
                types[item] = 'S'
        return types

## In the previous experiment with Italian data, the dictionary ''type'' was
## in the form of {expressionType : expressionFreq}
## Here, following the division in VNC-Tokens dataset we have the dictionary 
## in the form of: { expressionType : D or T*}
## D, when the expression is in Development portion of Cook's VNC-Tokens data
## T, when the expression is in Test portion of Cook's VNC-Tokens data

	 
# seqs is a list of tuples of this form: 
# (expression occurrence with its neighboring words, expression type, MWE label)

"""custom (type-aware) 10-fold cross-validation in the same format as sklearn's cross-val"""
def typeaware_cross_val(classifier, X, y, seqs): # my ten folds for cross-validation
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

"""this evaluation function is used only for the CRF classifier and
   evaluates only the beginning of VN expression"""
def evaluate(pred, yTest):
        acc = 0
        for i in range(len(yTest)):
                if 'B' in yTest[i] and 'B' in pred[i] and yTest[i].index('B')==pred[i].index('B'):
                        acc+=1
                elif 'B' not in yTest[i] and 'B' not in pred[i]:
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
                        elif seqs[j][1] in types:
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

"""standard random cross-validation for CRF"""
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

                y_pred = classifier.predict(X_test)
                sc = evaluate(y_pred, y_test)
                scores.append(sc)
        return np.array(scores)


"""type-aware cross-validation for the NLTK formated classifiers (we use Naive bayes here)"""	
def type_aware_cross_val_nltk(nltkClassifier, data, seqs):
        scores = []
        types = list(readTypes().keys())
        random.shuffle(types)
        testTypeSize = int(len(types)/10)
        for i in range(10):
                train_data, test_data = [], []
                for j in range(len(seqs)):
                        if seqs[j][1] in types[testTypeSize*i:testTypeSize*(i+1)]:
                                test_data.append(data[j])
                        elif seqs[j][1] in types:
                                train_data.append(data[j])                               
                classifier = nltk.NaiveBayesClassifier.train(train_data)
                scores.append(nltk.classify.util.accuracy(classifier, test_data))

        return np.array(scores)
    
