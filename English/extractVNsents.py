"""
This is the code to retrieve sentences from the BNC corpus (XML edition) relevant to the VNC-Tokens dataset.
The sentences are saved in a pickle file to be used as input data for our classifiers
"""
import nltk
import pickle

""" This fuction is an xml reader to retrive sentences of BNC based on their indices"""
def getSentence(bncXML, num):
          ### read XML sentences
          elemList = []
          for elem in bncXML.iter():
                    if elem.tag == 's':
                              elemList.append(elem) 
          sent = []
          for elem in elemList:
                    if elem.attrib['n']== num:
                              for child in elem.iter():
                                        if child.tag == 'w':
                                                  if 'hw' in child.attrib:
                                                            sent.append((child.text.strip(), child.attrib['hw'], child.attrib['pos']))
                                                  else:
                                                            sent.append((child.text.strip(), child.text.strip(), child.text.strip()))
          return sent
                    
### Read and put VNC-Tokens entries in a list
### to be used as queries to extract sentences from BNC
VNC_tokens = []
### PROVIDE THE PATH TO VNC-Tokens Dataset BEFORE RUNNING THE CODE 
with open("PATH/TO/VNC-Tokens Dataset") as f:
          lines = f.readlines()
          for l in lines:
                    l1 = l.split()
                    VNC_tokens.append((l1[2], l1[3], l1[1], l1[0]))

tagged_sents = []   # list of sentences
                    # each sentences is a tuple of the form: (sentence list, containing VNC, contating VNC Lable)
                    # A sentence list is a list of tuples of the form (word, word lemma, word pos)

### PROVIDE THE PATH TO BNC corpus (XML format) BEFORE RUNNING THE CODE
my_BNC_cr = nltk.corpus.reader.BNCCorpusReader('PATH/TO/BNC Corpus', '.*\.xml')

### We cannot use BNCCorpusReader.sents for this task, because the line numbers in the XML file and
### also given in VNC-Tokens are different from indices in sents.
### some of the line numbers do not exist in BNC xml files!
### So, we run our own xml reader (getSentence) on earch BNC file

for tag in VNC_tokens:
          xmlText = my_BNC_cr.xml(tag[0]+'.xml')
          sent = getSentence(xmlText, tag[1])                    
          tagged_sents.append((sent, tag[2], tag[3]))

with open("taggedSentsBNC.pkl",'wb') as f:
          pickle.dump(tagged_sents, f)
          
'''
seqs = pickle.load(open("sequencesBNC.pkl",'rb'))
print(len(seqs))
'''
