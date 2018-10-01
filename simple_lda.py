import tensorflow as tf
import edward as ed
import numpy as np
from edward.models import *

D = 2  # number of documents
N = [0] * D  # words per doc
K = 10  # number of topics
V = 1161 # vocabulary size
T = 1000

wordIds = [None] * D

with open("DataPreprocess/nipstxt/nips12/doc_wordID0003.txt") as f:
    wordIds[0] = list(map(int, f.readline().split()))
    N[0] = len(wordIds[0])
    wordIds[0] = np.array(wordIds[0])

print("load docwordID0003.txt finished")

with open("DataPreprocess/nipstxt/nips12/doc_wordID0010.txt") as f:
    wordIds[1] = list(map(int, f.readline().split()))
    N[1] = len(wordIds[1])
    wordIds[1] = np.array(wordIds[1])
print("load docwordID0010.txt finished")

# efficient lda model
alpha = tf.zeros([K]) + 0.1
#theta = Dirichlet(alpha, sample_shape=D)
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

#qtheta = Empirical(tf.Variable(tf.ones([T, D, K]) / K))
qtheta = [None] * D
qbeta = Empirical(tf.Variable(tf.ones([T, K, V]) / V))
qz = [None] * D
latent_vars = {}
#latent_vars[theta] = qtheta
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
#theta_cond = ed.complete_conditional(theta)
#proposal_vars_dict[theta] = theta_cond
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
