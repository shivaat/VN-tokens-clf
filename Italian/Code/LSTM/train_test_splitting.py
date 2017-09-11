"""random and type-aware train-test splitting for neural-nased classifiers"""
import pickle
import random

# the typeDist file contains each expression's type, its frequency and the percentage of time 
# the expression was annotated by '1' (as opposed to '0')
def readTypes():
	types = {}
	with open("../../Data/typeDist.txt", "r", encoding="utf8") as f:
		lines = f.readlines()
		for l in lines[15:]:
			l1 = l.split('\t')
			# this weeds out expressions with a frequency of lower than 3 
			if int(l1[1])>2:
				types[l1[0]] = float(l1[2])
	return list(types.keys())

"""type-aware 10-fold splitting to be used for cross-validation"""
def train_test(seqs): 
	train10 = []
	test10 = []
	types = readTypes()
	random.shuffle(types)
	testTypeSize = int(len(types)/10)
	for i in range(10):
		train = []
		test = []
		for j in range(len(seqs)):
			if seqs[j][1] in types[testTypeSize*i:testTypeSize*(i+1)]:
				test.append(seqs[j])
			elif seqs[j][1] in types:
				train.append(seqs[j])
		train10.append(train)
		test10.append(test)
	return train10, test10

"""regular 10-fold splitting to be used for cross-validation"""
from sklearn.model_selection import KFold
from sklearn import cross_validation
def random_kfold(seqs):
 	types = readTypes()
 	selected_seqs = [s for s in seqs if s[1] in types]
 	random.shuffle(selected_seqs)
 	train10 = []
 	test10 = []
 	cv = cross_validation.KFold(len(selected_seqs), n_folds=10, shuffle=False, random_state=None) 
 	for traincv, testcv in cv:
                train10.append([selected_seqs[i] for i in traincv])
                test10.append([selected_seqs[i] for i in testcv])
 	return train10, test10
