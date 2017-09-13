This folder includes the data and code for classifying the sentences in VNC-Tokens data set, based on whether they contain Verb-Noun MWEs or not.
The format of English VNC-tokens data set is a bit different from our Italian and Spanish data; hence the slight difference in code.

In this section, <b>first the script extractVNsents.py should be run</b> using VNC-tokens data set and the BNC corpus, which will result in creation of the file taggedSentsBNC.pkl. This file is then used by other scripts as input data for classification. 
