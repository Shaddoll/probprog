import numpy as np
import tensorflow as tf
import edward as ed
import glob
from models.lda import GaussianLDA

datafile = "DataPreprocess/nips12we25/short_wordembed_*.txt"
txt_files = glob.glob(datafile)
D = len(txt_files)  # number of documents
print("number of documents, D: {}".format(D))
N = [0] * D  # words per doc
K = 5  # number of topics
T = 500
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
"""
IdtoWord = {}
vocab = set()
with open("DataPreprocess/word_vectors.txt") as f:
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
print("load nounToID_50.txt finished")
"""


model = GaussianLDA(K, D, N, nu)
print("model constructed")
model.klqp(wordIds, S, T)
