import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import glob


def doc_cleaner(doc):
    """
    Clean and preprocess a document.

    1. Use regex to remove all special characters (only keep letters)
    2. Make strings to lower case and tokenize / word split reviews
    3. Remove English stopwords
    :param doc: document string
    :return: cleaned document string
    """
    doc = re.sub("[^a-zA-Z]", " ", doc)
    doc = doc.lower().split()
    eng_stopwords = stopwords.words("english")
    doc = [w for w in doc if w not in eng_stopwords]
    ps = PorterStemmer()
    ps_stems = []
    for word in doc:
        ps_stems.append(ps.stem(word))
    res = ' '.join(ps_stems)
    return res


# Clean words in each document
txt_files = glob.glob("nipstxt/nips12/*.txt")
# txt_files = glob.glob("nips08/*.txt")
for file in sorted(txt_files):
    doc = []
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        body = False
        for line in f:
            line = line.strip()
            if line == 'Abstract':
                body = True
            # Extract the Abstract part of each document
            if line == "1 INTRODUCTION" or line == "1 Introduction":
                body = False
            if body:
                if line[-1] == '-':
                    line = line.strip('-')
                    doc.append(line)
                else:
                    line = line + ' '
                    doc.append(line)

    doc = doc_cleaner("".join(doc))
    with open('nipstxt/nips12/'+'short_12_'+file[-8:-4]+'.txt', 'a') as f2:
        f2.write(doc)
        f2.close()

txt_files_clean = glob.glob("nipstxt/nips12/short_*.txt")
for file in sorted(txt_files_clean):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        for line in f:
            doc_words = line.split()
            if len(doc_words) > 300:
                print(file)
# Generate a vocabulary set that contains unique words
# txt_files_clean = glob.glob("nipstxt/nips12/clean_*.txt")
txt_files_clean = glob.glob("nips2yabs/short_*.txt")
vocabulary = set()
for file in sorted(txt_files_clean):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        for line in f:
            doc_words = line.split()
            for word in doc_words:
                vocabulary.add(word)
w_id = 0
wordToID = {}
for key in vocabulary:
    wordToID[key] = id
    w_id += 1
print("The size of Vocabulary: ")
print(len(vocabulary))
# Change document with word IDs
txt_files_clean_2 = glob.glob("nips2y/clean_*.txt")
for file in sorted(txt_files_clean_2):
    with open(file, 'rt', encoding="ISO-8859-1") as f:
        doc_wordID = ''
        for line in f:
            doc_words = line.split()
            for word in doc_words:
                if word in wordToID:
                    doc_wordID += str(wordToID[word])
                    doc_wordID += ' '
                else:
                    doc_wordID += 'n'
                    doc_wordID += ' '
        with open('nips2y/'+'doc_clean_wID'+file[-11:-4]+'.txt', 'a') as f2:
            f2.write(doc_wordID)
            f2.close()
