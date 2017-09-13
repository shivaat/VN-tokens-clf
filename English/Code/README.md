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

