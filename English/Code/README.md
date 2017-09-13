This directory includes the following files:

typeAwareCV.py
  	
	This file includes functions that perform different forms of cross-validation.
	The fuctions are used by vncCompareClassifiersCustomKfold.py and vncCompareClassifiersRegularKfold.py 
  
vncCompareClassifiersCustomKfold.py
  
	This part of code compares the performances of different traditional classifiers:
  	Logistic Regression, Decision Tree, Random Forest, Multiple Layer Perceptron and SVM.
  	The performances are computed using custom typeaware cross-validation
  
vncCompareClassifiersRegularKfold.py
  	
	This part of code compares the performances of different traditional classifiers:
  	Logistic Regression, Decision Tree, Random Forest, Multiple Layer Perceptron and SVM.
  	The performances are computed using regular cross-validation
	
CRF.py

	This file contains functions for the CRF classfier
	
train_test_splitting.py

	This file includes the fuctions for random and type-aware train-test splitting for neural-based classifiers
	The fuctions are used by deep_compare_VNC.py
	
deep_compare_VNC.py

	Using this script we compared performances of several different neural network classifiers.
	The best performing model is the ConvNet+LSTM
	The results can be reported both using regular and type-aware cross-validation.
	Relevant lines should be commented/uncommented
	
init_weights.h5

	This file contains the initial weights to be used by the neural based classifiers in deep_compare_VNC.py

