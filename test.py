import numpy as np
import tensorflow as tf
import edward as ed
import glob
import pickle
from models.lda import GaussianLDA
import time

t1 = time.time()
datafile = "DataPreprocess/nips12we25/short_wordembed_*.txt"
txt_files = glob.glob(datafile)
D = len(txt_files)  # number of documents
print("number of documents, D: {}".format(D))
N = [0] * D  # words per doc
K = 9  # number of topics
T = 1500
S = 5
wordIds = [None] * D
count = 0  # count number of documents
for file in (txt_files):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        vec = []
        for line in f:
            vec.append(list(map(float, line.split())))
        N[count] = len(vec)
        wordIds[count] = vec
    print("load" + file + "finished")
    count += 1
nu = len(wordIds[0][0])
print("dimension:", nu)

wordToId = dict()
tokens = []
with open("DataPreprocess/wordToID_short_12.txt") as f:
    for line in f:
        line = line.split()
        wordToId[line[0]] = len(tokens)
        tokens.append(line[0])
V = len(tokens)  # vocabulary size21
print("vocab size is {}".format(V))
print("load wordId finished")

wordVec = [None] * V
with open("DataPreprocess/word_vectors_25.txt") as f:
    for line in f:
        line = line.split()
        if line[0] in wordToId:
            wordVec[wordToId[line[0]]] = list(map(float, line[1:]))
print("load word embeddings finished")
D=50

model = GaussianLDA(K, D, N, nu)
print("model constructed")
model.klqp(wordIds, S, T, wordVec)
print("inference finished")
print(time.time() - t1)
model.getTopWords(wordVec, tokens)
print("get top words finished")
print(time.time() - t1)
comatrix = pickle.load(open("DataPreprocess/comatrix1y.pickle", "rb"))
model.getPMI(comatrix)
print(time.time() - t1)
