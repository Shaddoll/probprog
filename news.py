import pickle
from models.lda import LDA


wordIds = pickle.load(open("toy.dat", "rb"))
wordIds = wordIds[:50]
tokens = list(pickle.load(open("vocab.dat", "rb")))
K = 20
V = len(tokens)
D = len(wordIds)
N = [len(wordIds[d]) for d in range(D)]
S = 100
T = 500
model = LDA(K, V, D, N)
model.collapsed(wordIds, S, T, tokens)
