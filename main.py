import numpy as np
import glob
from models.lda import LDA
import tensorflow as tf

"""
datafile = "DataPreprocess/nipstxt/nipstoy20/doc_wordID_short*.txt"
txt_files = glob.glob(datafile)
D = len(txt_files)  # number of documents
print("number of documents, D: {}".format(D))
N = [0] * D  # words per doc
K = 5  # number of topics
T = 100
S = 10
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
with open("DataPreprocess/wordToIDtoy20.txt") as f:
    for line in f:
        line = line.split()
        IdtoWord[int(line[1])] = line[0]
        vocab.add(line[0])
V = len(vocab)  # vocabulary size21
tokens = [None] * V
for key in IdtoWord:
    tokens[key] = IdtoWord[key]
print("vocab size is {}".format(V))
print("load wordToIDtoy.txt finished")
"""
import pickle
wordIds = pickle.load(open("toy.dat", "rb"))
tokens = list(pickle.load(open("vocab.dat", "rb")))
K = 5
V = len(tokens)
D = len(wordIds)
N = [len(wordIds[d]) for d in range(D)]
S = 10
T = 100

model = LDA(K, V, D, N)
print("model constructed")
model.gibbs(wordIds, S, T)
#model.klqp(wordIds, S, T)
model.criticize(tokens)