This folder includes the data and code for classifying the sentences in VNC-Tokens data set, based on whether they contain Verb-Noun MWEs or not.
The format of English VNC-tokens data set is a bit different from our Italian and Spanish data; hence the slight difference in codes.

In this section, first the code extractVNsents.py should be run using VNC-tokens data set and the BNC corpus, in order to create the file taggedSentsBNC.pkl. The file taggedSentsBNC.pkl is then used by other codes as data instances to be classified.
