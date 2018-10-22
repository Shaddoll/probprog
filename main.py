import numpy as np
import glob
from models.lda import LDA
import tensorflow as tf


datafile = "DataPreprocess/nipstxt/nipstoy20/doc_wordID_short*.txt"
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

model = LDA(K, V, D, N)
print("model constructed")
model.gibbs(wordIds, S, T)
#model.klqp(wordIds, S, T)
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