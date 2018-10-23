import numpy as np
import glob
from models.lda import LDA
import tensorflow as tf


<<<<<<< HEAD
datafile = "DataPreprocess/nipstxt/nipstoy30/doc_wordID_short*.txt"
=======
datafile = "DataPreprocess/nipstxt/nipstoy20/doc_wordID_short*.txt"
>>>>>>> a87e1559b9473e00597348171c7383eade406ba8
txt_files = glob.glob(datafile)
D = len(txt_files)  # number of documents
print("number of documents, D: {}".format(D))
N = [0] * D  # words per doc
K = 5  # number of topics
T = 2000
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
<<<<<<< HEAD
with open("DataPreprocess/wordToIDtoy30.txt") as f:
=======
with open("DataPreprocess/wordToIDtoy20.txt") as f:
>>>>>>> a87e1559b9473e00597348171c7383eade406ba8
    for line in f:
        line = line.split()
        IdtoWord[int(line[1])] = line[0]
        vocab.add(line[0])
V = len(vocab)  # vocabulary size21
tokens = [None] * V
for key in IdtoWord:
    tokens[key] = IdtoWord[key]
print("vocab size is {}".format(V))
print("load wordToIDtoy30.txt finished")

model = LDA(K, V, D, N)
print("model constructed")
model.gibbs(wordIds, S, T)
<<<<<<< HEAD
#model.klqp(wordIds, T)
tokens = [None] * V
for key in IdtoWord:
    tokens[key] = IdtoWord[key]
=======
#model.klqp(wordIds, S, T)
>>>>>>> a87e1559b9473e00597348171c7383eade406ba8
model.criticize(tokens)

def assert_close(
    x, y, data=None, summarize=None, message=None, name="assert_close"):
    message = message or ""
    x = tf.convert_to_tensor(x, name="x")
    y = tf.convert_to_tensor(y, name="y")
    if data is None:
        data = [
            message,
            "Condition x ~= y did not hold element-wise: x = ", x.name, x, "y = ",
            y.name, y
        ]
    if x.dtype.is_integer:
        return tf.assert_equal(
            x, y, data=data, summarize=summarize, message=message, name=name)
    with tf.name_scope(name, "assert_close", [x, y, data]):
        tol = np.finfo(x.dtype.as_numpy_dtype).eps
        condition = tf.reduce_all(tf.less_equal(tf.abs(x - y), tol))
        return tf.Assert(
            condition, data, summarize=summarize)