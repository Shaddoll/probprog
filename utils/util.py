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
