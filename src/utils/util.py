import numpy as np
import math


def reverseMap(tokens):
    result = dict()
    n = len(tokens)
    for i in range(n):
        result[tokens[i]] = i
    return result


def get_ngrams(sequence, n):
    if n <= 0:
        return []
    ngrams = []
    ngram = ('START',) * n
    ngrams.append(ngram)
    for word in sequence:
        ngram = ngram[1:] + (word, )
        ngrams.append(ngram)
    ngram = ngram[1:] + ('STOP', )
    ngrams.append(ngram)
    return ngrams


def cooccurence(corpus, vocabs, window_size):
    n = len(vocabs) + 1
    result = np.zeros((n, n))
    for doc in corpus:
        n = len(doc)
        for i in range(n):
            for j in range(i + 1, min(n, i + window_size)):
                result[doc[i]][doc[j]] += 1.0
    return result


def pmi(comatrix, words, wordToId=None):
    N = comatrix.sum()
    n = len(words)
    cnt = 0.0
    result = 0.0
    for i in range(n):
        if wordToId is None:
            a = words[i]
        else:
            a = wordToId[words[i]]
        for j in range(i + 1, n):
            if wordToId is None:
                b = words[j]
            else:
                b = wordToId[words[j]]
            if comatrix[a][b] > 0:
                result += math.log2(comatrix[a][b])
                result += math.log2(N)
                result -= math.log2(comatrix[a].sum())
                result -= math.log2(comatrix[:, b].sum())
                cnt += 1
            if comatrix[b][a] > 0:
                result += math.log2(comatrix[b][a])
                result += math.log2(N)
                result -= math.log2(comatrix[b].sum())
                result -= math.log2(comatrix[:, a].sum())
                cnt += 1
    print(cnt)
    return result / cnt
