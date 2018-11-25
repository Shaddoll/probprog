import numpy as np
import glob
from models.lda import LDA
import time
import pickle

t1 = time.time()

# 20 documents with abstract only
# datafile = "DataPreprocess/nipstxt/nipstoy20/doc_wordID_short*.txt"
# All documents with nouns
# datafile = "DataPreprocess/nipstxt/nips12nouns/doc_nounID*.txt"
# 50 documents with nouns in the abstract
# datafile = "DataPreprocess/nipstxt/nipstoy50/abstract_nounID*.txt"
# all documents with nouns in the abstract
# datafile = "DataPreprocess/nipstxt/nipstoyall/abstract_nounID*.txt"
# datafile = "DataPreprocess/nipstxt/nipstoy/short_wordID*.txt"
datafile = "DataPreprocess/nipstxt/nipstoy20/doc_short_wID*.txt"
txt_files = glob.glob(datafile)
D = len(txt_files)  # number of documents
print("number of documents, D: {}".format(D))
N = [0] * D  # words per doc
# K = 10  # number of topics
K = 9  # number of topics
T = 300
S = 200
wordIds = [None] * D
count = 0  # count number of documents
for file in (txt_files):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        wordIds[count] = list(map(int, f.readline().split()))
        N[count] = len(wordIds[count])
        wordIds[count] = np.array(wordIds[count]).astype('int32')
    print("load" + file + "finished")
    count += 1
IdtoWord = {}
vocab = set()
# with open("DataPreprocess/nounToID.txt") as f:
# with open("DataPreprocess/nounToID_50.txt") as f:
# with open("DataPreprocess/nounToID_abstract.txt") as f:
with open("DataPreprocess/wordToID_short_12_20.txt") as f:
    for line in f:
        line = line.split()
        IdtoWord[int(line[1])] = line[0]
        vocab.add(line[0])
V = len(vocab)  # vocabulary size21
tokens = [None] * V
for key in IdtoWord:
    tokens[key] = IdtoWord[key]
print("vocab size is {}".format(V))
# print("load wordToIDtoy.txt finished")
print("load wordToID_3y.txt finished")


model = LDA(K, V, D, N)
print("model constructed")
model.collapsed(wordIds, S, T, tokens)
print("inference finished")
print(time.time() - t1)
model.getTopWords(tokens)
comatrix = pickle.load(open("20abstract1yearAll.pickle", "rb"))
model.getPMI(comatrix)
