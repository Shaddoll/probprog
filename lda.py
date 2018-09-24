import tensorflow as tf
import numpy as np
#from edward.models import Categorical, Dirichlet, ParamMixture, WishartFull, MultivariateNormalFullCovariance, Empirical
from edward.models import *
from hyperspherical_vae.distributions import VonMisesFisher
from gem import Gem

def __init__(self, *args, **kwargs):
    RandomVariable.__init__(self, *args, **kwargs)

__init__.__doc__ = VonMisesFisher.__init__.__doc__
_params = {'__doc__': VonMisesFisher.__doc__,
           '__init__': __init__}
_globals = globals()
_globals['VonMisesFisher'] = type('VonMisesFisher', (RandomVariable, VonMisesFisher), _params)

D = 4  # number of documents
N = [11502, 213, 1523, 1351]  # words per doc
K = 10  # number of topics
V = 100000  # vocabulary size
T = 10000

# efficient lda model
alpha = tf.zeros([D, K]) + 0.1
theta = Dirichlet(alpha)
yita = tf.zeros([K, V]) + 0.05
beta = Dirichlet(yita)
z = [None] * D
w = [None] * D
for d in range(D):
    w[d] = ParamMixture(mixing_weights=theta[d],
                        component_params={'probs': beta},
                        component_dist=Categorical,
                        sample_shape=N[d],
                        validate_args=True)
    z[d] = w[d].cat

qtheta = Empirical(tf.Variable(tf.ones([T, D, K]) / K))
qbeta = Empirical(tf.Variable(tf.ones([T, K, V]) / V))
qz = [None] * D
latent_vars = {}
latent_vars[theta] = qtheta
latent_vars[beta] = qbeta
training_data = {}
for d in range(D):
    qz[d] = Empirical(tf.Variable(tf.zeros([T, N[d]])), dtype=tf.int32)
    latent_vars[z[d]] = qz[d]
    training_data[w[d]] =

inference = ed.Gibbs(latent_vars, training_data)
inference.initialize(n_iter=T)
tf.global_variables_initializer().run()
for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
inference.finalize()

nu = 50
mu_0 = [0.0] * nu
sigma_0 = tf.eye(nu)
sigma = WishartFull(df=nu, scale=sigma_0, sample_shape=K)
mu = MultivariateNormalFullCovariance(loc=mu_0, covariance_matrix=tf.matrix_inverse(sigma))

alpha = tf.zeros([D, K]) + 0.1
theta = Dirichlet(alpha)
w = [None] * D
z = [None] * D
for d in range(D):
    w[d] = ParamMixture(mixing_weights=theta[d],
                        component_params={'loc': mu, 'covariance_matrix': tf.matrix_inverse(sigma)},
                        component_dist=MultivariateNormalFullCovariance,
                        sample_shape=N[d],
                        validate_args=True)
    z[d] = w[d].cat

qtheta = Empirical(tf.Variable(tf.ones([T, D, K]) / K))
qsigma = Empirical(tf.Variable(tf.ones([T, K, nu, nu])))
qmu = Empirical(tf.Variable(tf.ones([T, K, nu])))
qz = [None] * D
for d in range(D):
    qz[d] = Empirical(tf.Variable(tf.zeros([T, N[d]])), dtype=tf.int32)


# sHDP
# GEM(gamma) = beta(1, gamma)
alpha = 1.
gamma = 2.
beta = TransformedDistribution(
  distribution=Beta([1.] * K, [gamma] * K),
  bijector=Gem(),
  name="GemTransformedDistribution")
pi = DirichletProcess(alpha, beta, sample_shape=D)

d = np.random.rand(50,)
d = d/np.linalg.norm(d)
mu_0 = d.astype('float32')
c_0 = 1.
m = 2.
sigma = 0.25
kappa = TransformedDistribution(
    distribution=Normal(loc=m, scale=sigma),
    bijector=tf.contrib.distributions.bijectors.Exp(),
    sample_shape=K,
    name="LogNormalTransformedDistribution")

mu = VonMisesFisher(np.array([mu_0] * K), [[c_0]] * K)
w = [None] * D
z = [None] * D
w = [None] * D
z = [None] * D
for d in range(D):
    w[d] = ParamMixture(mixing_weights=pi[d],
                        component_params={'loc': mu, 'scale': kappa},
                        component_dist=VonMisesFisher,
                        sample_shape=N[d],
                        validate_args=True)
    z[d] = w[d].cat
