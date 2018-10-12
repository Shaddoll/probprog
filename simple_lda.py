import tensorflow as tf
import edward as ed
import numpy as np
# from heapq import nlargest



from edward.models import *

D = 2  # number of documents
N = [0] * D  # words per doc
K = 2  # number of topics
V = 1161 # vocabulary size21
T = 100

wordIds = [None] * D

with open("DataPreprocess/nipstxt/nips12/doc_wordID_short0003.txt") as f:
    wordIds[0] = list(map(int, f.readline().split()))
    N[0] = len(wordIds[0])
    x = wordIds[0]
    wordIds[0] = np.array(wordIds[0])

print("load docwordID0003.txt finished")

with open("DataPreprocess/nipstxt/nips12/doc_wordID_short0010.txt") as f:
    wordIds[1] = list(map(int, f.readline().split()))
    N[1] = len(wordIds[1])
    y = wordIds[1]
    wordIds[1] = np.array(wordIds[1])

IdtoWord = {}
with open("DataPreprocess/wordToIDshort.txt") as f:
    for line in f:
        line = line.split()
        IdtoWord[int(line[1])] = line[0]
# print (IdtoWord)

x = set(x)
y = set(y)
vocab = x.union(y)
V = len(x.union(y))
print (V)
print("load docwordID0010.txt finished")

# efficient lda model
alpha = tf.zeros([K]) + 0.1
theta = [None] * D
yita = tf.zeros([V]) + 0.05
beta = Dirichlet(yita, sample_shape=K)
z = [None] * D
w = [None] * D
for d in range(D):
    theta[d] = Dirichlet(alpha)
    w[d] = ParamMixture(mixing_weights=theta[d],
                        component_params={'probs': beta},
                        component_dist=Categorical,
                        sample_shape=N[d],
                        validate_args=True)
    z[d] = w[d].cat

print("model constructed")

qtheta = [None] * D
qbeta = Empirical(tf.Variable(tf.ones([T, K, V]) / V))
qz = [None] * D
latent_vars = {}
latent_vars[beta] = qbeta
training_data = {}
for d in range(D):
    qtheta[d] = Empirical(tf.Variable(tf.ones([T, K]) / K))
    latent_vars[theta[d]] = qtheta[d]
    qz[d] = Empirical(tf.Variable(tf.zeros([T, N[d]], dtype=tf.int32)))
    latent_vars[z[d]] = qz[d]
    training_data[w[d]] = wordIds[d]


proposal_vars_dict = {}
beta_cond = ed.complete_conditional(beta)
proposal_vars_dict[beta] = beta_cond
theta_cond = [None] * D
z_cond = [None] * D
for d in range(D):
    theta_cond[d] = ed.complete_conditional(theta[d])
    proposal_vars_dict[theta[d]] = theta_cond[d]
    z_cond[d] = ed.complete_conditional(z[d])
    proposal_vars_dict[z[d]] = z_cond[d]



inference = ed.Gibbs(latent_vars, proposal_vars_dict, data=training_data)
inference.initialize(n_iter=T)
tf.global_variables_initializer().run()
for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
inference.finalize()

# sample beta
qbeta_sample = latent_vars[beta].params[-1].eval()


prob = [None] * K
for k in range(K):
    prob[k] = qbeta_sample[k, :]
    print (len(prob[k]))
#
tokens = [None] * V
for i, id in enumerate(list(vocab)):
    # if id not in IdtoWord:
    #     print (id)
    tokens[i] = IdtoWord[id]
# #
tokens_probs =[None] * K
for k in range(K):
    tokens_probs[k] =dict((t, p) for t, p in zip(tokens, prob[k]))
    newdict = sorted(tokens_probs[k], key=tokens_probs[k].get, reverse=True)[:15]
    print ('topic %d'%k)
    for word in newdict:
        print (word)