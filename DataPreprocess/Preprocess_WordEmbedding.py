import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import glob
from gensim.models import word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import numpy as np


def doc_cleaner(doc):
    """
    Clean and preprocess a document.

    1. Use regex to remove all special characters (only keep letters)
    2. Make strings to lower case and tokenize / word split reviews
    3. Remove English stopwords
    :param doc: document string
    :return: a list of cleaned words
    """
    doc = re.sub("[^a-zA-Z]", " ", doc)
    doc = doc.lower().split()
    eng_stopwords = stopwords.words("english")
    doc = [w for w in doc if w not in eng_stopwords]
    ps = PorterStemmer()
    ps_stems = []
    for word in doc:
        ps_stems.append(ps.stem(word))
    return(ps_stems)


# Clean words in each document while keep every sentences
txt_files = glob.glob("nips12we25/*.txt")
corpus = []
for file in sorted(txt_files):
    doc = []
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        body = False
        for line in f:
            line = line.strip()
            if line == 'Abstract':
                body = True
            if line == 'References':
                body = False
            if body:
                if line[-1] == '-':
                    line = line.strip('-')
                    doc.append(line)
                else:
                    line += ' '
                    doc.append(line)
    doc = ''.join(doc)
    doc += ' '
    corpus.append(doc)
corpus = ''.join(corpus)
corpus = corpus.split('.')
for i, line in enumerate(corpus):
    corpus[i] = doc_cleaner(line)
# # Set values for various parameters
# num_features = 25   # Word vector dimensionality
# min_word_count = 0   # ignore all words with total frequency lower than this
# num_workers = 4       # Number of threads to run in parallel
# context = 10          # Context window size
# Initialize and train the model (this will take some time)
print("Training word2vec model... ")
model = word2vec.Word2Vec(corpus, workers=4, size=25, min_count=0, window=10)
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "25dim_0minwords_10context"
model.save(model_name)
vocab_tmp = list(model.wv.vocab)
print('Vocab length:', len(vocab_tmp))
wvec = model.wv
fname = get_tmpfile("vectors.kv")
wvec.save(fname)
model.wv.save_word2vec_format('word_vectors_25.txt', binary=False)
# Create word embeddings for document words
wvec = KeyedVectors.load(fname, mmap='r')
txt_files_clean = glob.glob("nipstxt/nips12/short_*.txt")
for file in sorted(txt_files_clean):
    dw_embed = np.array([0]*25)
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        for line in f:
            doc_words = line.split()
            for word in doc_words:
                doc_word_embed = np.vstack([dw_embed, wvec[word]])
    np.savetxt('short_we_'+file[-8:-4]+'.txt', dw_embed[1:, :], delimiter=' ')
