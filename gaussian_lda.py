import tensorflow as tf
import edward as ed
import numpy as np
from edward.models import *

D = 2  # number of documents
N = [100] * D  # words per doc
K = 10  # number of topics
V = 1161 # vocabulary size
T = 30


nu = 50
mu_0 = [0.0] * nu
sigma_0 = tf.eye(nu)
sigma = WishartFull(df=nu, scale=sigma_0, sample_shape=K)
mu = MultivariateNormalFullCovariance(loc=mu_0, covariance_matrix=tf.matrix_inverse(sigma))


alpha = tf.zeros([K]) + 0.1
theta = [None] * D
w = [None] * D
z = [None] * D
for d in range(D):
    theta[d] = Dirichlet(alpha)
    w[d] = ParamMixture(mixing_weights=theta[d],
                        component_params={'loc': mu, 'covariance_matrix': tf.matrix_inverse(sigma)},
                        component_dist=MultivariateNormalFullCovariance,
                        sample_shape=N[d],
                        validate_args=True)
    z[d] = w[d].cat

latent_vars = {}
qtheta = [None] * D
qsigma = Empirical(tf.Variable(tf.ones([T, K, nu, nu])))
qmu = Empirical(tf.Variable(tf.ones([T, K, nu])))
qz = [None] * D
for d in range(D):
    qtheta[d] = Empirical(tf.Variable(tf.ones([T, K]) / K))
    qz[d] = Empirical(tf.Variable(tf.zeros([T, N[d]])), dtype=tf.int32)
