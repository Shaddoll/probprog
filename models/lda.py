import tensorflow as tf
import edward as ed
import numpy as np
from utils import util
from edward.models import Dirichlet, ParamMixture, Categorical, Empirical, \
                          WishartCholesky, MultivariateNormalTriL, \
                          Normal, Mixture, InverseGamma, MultivariateNormalDiag
ds = tf.contrib.distributions


class LDA(object):
    def __init__(self, K, V, D, N):
        self.K = K  # number of topics
        self.V = V  # vocabulary size
        self.D = D  # number of documents
        self.N = N  # number of words of each document
        self.alpha = alpha = tf.zeros([K]) + 0.1
        self.yita = yita = tf.zeros([V]) + 0.01
        self.theta = [None] * D
        self.beta = Dirichlet(yita, sample_shape=K)
        self.z = [None] * D
        self.w = [None] * D
        temp = self.beta
        for d in range(D):
            self.theta[d] = Dirichlet(alpha)
            self.w[d] = ParamMixture(mixing_weights=self.theta[d],
                                     component_params={'probs': temp},
                                     component_dist=Categorical,
                                     sample_shape=N[d],
                                     validate_args=False)
            self.z[d] = self.w[d].cat

    def __run_inference__(self, T, S=None):
        tf.global_variables_initializer().run()
        for n in range(self.inference.n_iter):
            info_dict = self.inference.update()
            self.inference.print_progress(info_dict)
        self.inference.finalize()

    def gibbs(self, wordIds, S, T):
        K = self.K
        V = self.V
        D = self.D
        N = self.N
        latent_vars = {}
        training_data = {}
        qbeta = Empirical(tf.Variable(tf.zeros([S, K, V]) + 0.01))
        latent_vars[self.beta] = qbeta
        qtheta = [None] * D
        qz = [None] * D
        for d in range(D):
            qtheta[d] = Empirical(tf.Variable(tf.zeros([S, K]) + 0.1))
            latent_vars[self.theta[d]] = qtheta[d]
            qz[d] = Empirical(tf.Variable(tf.zeros([S, N[d]], dtype=tf.int32)))
            latent_vars[self.z[d]] = qz[d]
            training_data[self.w[d]] = wordIds[d]
        self.latent_vars = latent_vars
        self.inference = ed.Gibbs(latent_vars, data=training_data)
        print("gibbs setup finished")
        self.inference.initialize(n_iter=T, n_print=1)
        self.__run_inference__(T)
        self.qbeta_sample = qbeta.eval()

    def collapsed_gibbs(self, wordIds, S, T):
        K = self.K
        V = self.V
        D = self.D
        N = self.N
        latent_vars = {}
        training_data = {}
        qbeta = Empirical(tf.Variable(tf.zeros([S, K, V]) + 0.01))
        latent_vars[self.beta] = qbeta
        qtheta = [None] * D
        qz = [None] * D
        for d in range(D):
            qtheta[d] = Empirical(tf.Variable(tf.zeros([S, K]) + 0.1))
            latent_vars[self.theta[d]] = qtheta[d]
            qz[d] = Empirical(tf.Variable(tf.zeros([S, N[d]], dtype=tf.int32)))
            latent_vars[self.z[d]] = qz[d]
            training_data[self.w[d]] = wordIds[d]
        self.latent_vars = latent_vars
        proposal_vars = {}
        proposal_vars[self.beta] = ed.complete_conditional(self.beta)
        cond_set = set(self.w + self.z)
        for d in range(D):
            proposal_vars[self.theta[d]] = \
                ed.complete_conditional(self.theta[d])
            proposal_vars[self.z[d]] = \
                ed.complete_conditional(self.z[d], cond_set)
        self.inference = ed.Gibbs(latent_vars, proposal_vars, training_data)
        print("collapsed gibbs setup finished")
        self.inference.initialize(n_iter=T, n_print=1)
        print("initialize finished")
        self.__run_inference__(T)
        self.qbeta_sample = qbeta.eval()

    def collapsed(self, w, S, T, tokens):
        K = self.K
        V = self.V
        D = self.D
        N = self.N
        alpha = 0.1
        beta = 0.01
        _qz = [[0 for n in range(N[d])] for d in range(D)]
        _ndk = np.zeros((D, K), dtype='float32')
        _nvk = np.zeros((V, K), dtype='float32')
        for d in range(D):
            for n in range(N[d]):
                _qz[d][n] = z = np.random.randint(K)
                _ndk[d][z] += 1
                _nvk[w[d][n]][z] += 1
        self.qz = qz = [tf.Variable(_qz[d]) for d in range(D)]
        self.ndk = ndk = [tf.Variable(_ndk[d]) for d in range(D)]
        # self.ndk = ndk = tf.Variable(_ndk)
        self.nvk = nvk = [tf.Variable(_nvk[v]) for v in range(V)]
        # self.nvk = nvk = tf.Variable(_nvk)
        self.nk = nk = tf.reduce_sum(ndk, axis=0)
        qz_prob = [(ndk[d] + alpha) * tf.gather(nvk, w[d]) / (nk + V * beta) /
                   tf.reshape(tf.reduce_sum((ndk[d] + alpha) *
                                            tf.gather(nvk, w[d]) /
                                            (nk + V * beta), axis=-1),
                              (-1, 1)) for d in range(D)]
        self.qbeta = (tf.convert_to_tensor(nvk) + beta) / (nk + V * beta)
        self.update_ops = ops = [tf.no_op(), tf.no_op()]
        for d in range(D):
            for n in range(N[d]):
                with tf.control_dependencies([ops[-1], ops[-2]]):
                    oldk = qz[d][n]
                    ops.append(tf.scatter_sub(ndk[d], [oldk], [1]))
                    # ops.append(tf.scatter_nd_sub(ndk, [(d, oldk)], [1]))
                    ops.append(tf.scatter_sub(nvk[w[d][n]], [oldk], [1]))
                    # ops.append(tf.scatter_nd_sub(nvk, [(w[d][n], oldk)],[1]))
                # sequentially
                with tf.control_dependencies([ops[-1], ops[-2]]):
                    k = tf.multinomial(tf.log([qz_prob[d][n]]), 1,
                                       output_dtype=tf.int32)[0]
                    ops.append(tf.scatter_update(qz[d], [n], k))
                with tf.control_dependencies([ops[-1]]):
                    newk = qz[d][n]
                    ops.append(tf.scatter_add(ndk[d], [newk], [1]))
                    # ops.append(tf.scatter_nd_add(ndk, [(d, newk)], [1]))
                    ops.append(tf.scatter_add(nvk[w[d][n]], [newk], [1]))
                    # ops.append(tf.scatter_nd_add(nvk, [(w[d][n], newk)],[1]))
        sess = ed.get_session()
        # tf.global_variables_initializer().run()
        sess.run(tf.variables_initializer(qz + nvk + ndk))
        for _ in range(T):
            print("burnin:", _)
            sess.run(ops)
        qbeta_sample = np.zeros((V, K), dtype='float32')
        for _ in range(S):
            print("sample:", _)
            sess.run(ops)
            qbeta_sample += self.qbeta.eval()
        qbeta_sample /= S
        self.qbeta_sample = qbeta_sample

    def getTopWords(self, tokens):
        K = self.K
        V = self.V
        qbeta_sample = self.qbeta_sample
        prob = [None] * K
        for k in range(K):
            prob[k] = qbeta_sample[:, k]
        self.tokens_probs = tokens_probs = [None] * K
        self.top_words = [None] * K
        for k in range(K):
            tokens_probs[k] = dict((t, p) for t, p in zip(range(V), prob[k]))
            newdict = sorted(tokens_probs[k],
                             key=tokens_probs[k].get,
                             reverse=True)[:10]
            self.top_words[k] = newdict
            print('topic %d' % k)
            for Id in newdict:
                print(tokens[Id], tokens_probs[k][Id])

    def getPMI(self, comatrix):
        K = self.K
        self.pmis = pmis = [None] * K
        for k in range(K):
            pmis[k] = util.pmi(comatrix, self.top_words[k])
            print('topic %d pmi: %f' % (k, pmis[k]))


class GaussianLDA(object):
    def __init__(self, K, D, N, nu, use_param=False):
        self.K = K  # number of topics
        self.D = D  # number of documents
        self.N = N  # number of words of each document
        self.nu = nu
        self.alpha = alpha = tf.zeros([K]) + 0.1
        mu0 = tf.constant([0.0] * nu)
        sigma0 = tf.eye(nu)
        self.sigma = sigma = WishartCholesky(
            df=nu,
            scale=sigma0,
            cholesky_input_output_matrices=True,
            sample_shape=K)
        # sigma_inv = tf.matrix_inverse(sigma)
        self.mu = mu = Normal(mu0, tf.ones(nu), sample_shape=K)
        self.theta = theta = [None] * D
        self.z = z = [None] * D
        self.w = w = [None] * D
        for d in range(D):
            theta[d] = Dirichlet(alpha)
            if use_param:
                w[d] = ParamMixture(
                    mixing_weights=theta[d],
                    component_params={'loc': mu, 'scale_tril': sigma},
                    component_dist=MultivariateNormalTriL,
                    sample_shape=N[d])
                z[d] = w[d].cat
            else:
                z[d] = Categorical(probs=theta[d], sample_shape=N[d])
                components = [
                    MultivariateNormalTriL(loc=tf.gather(mu, k),
                                           scale_tril=tf.gather(sigma, k),
                                           sample_shape=N[d])
                    for k in range(K)]
                w[d] = Mixture(cat=z[d],
                               components=components,
                               sample_shape=N[d])

    def __run_inference__(self, T, S=None):
        tf.global_variables_initializer().run()
        for n in range(self.inference.n_iter):
            info_dict = self.inference.update()
            self.inference.print_progress(info_dict)
        self.inference.finalize()

    def klqp(self, docs, S, T, wordVec):
        K = self.K
        D = self.D
        nu = self.nu
        self.latent_vars = latent_vars = {}
        training_data = {}
        qmu = Normal(
            loc=tf.Variable(tf.random_normal([K, nu])),
            scale=tf.nn.softplus(tf.Variable(tf.zeros([K, nu]))))
        latent_vars[self.mu] = qmu
        qpsi0 = tf.Variable(tf.eye(nu, batch_shape=[K]))
        Ltril = tf.linalg.LinearOperatorLowerTriangular(
            ds.matrix_diag_transform(
                qpsi0,
                transform=tf.nn.softplus)).to_dense()
        qsigma = WishartCholesky(
            df=tf.ones([K])*nu,
            scale=Ltril,
            cholesky_input_output_matrices=True)
        latent_vars[self.sigma] = qsigma
        for d in range(D):
            training_data[self.w[d]] = docs[d]
        self.qmu = qmu
        # self.qsigma_inv = qsigma_inv = tf.matrix_inverse(qsigma)
        self.qw = MultivariateNormalTriL(loc=qmu, scale_tril=qsigma)
        V = len(wordVec)
        logprobs = [None] * V
        for i in range(V):
            logprobs[i] = self.qw.log_prob(wordVec[i])
        self.qbeta = tf.convert_to_tensor(logprobs)
        self.inference = ed.KLqp(latent_vars, data=training_data)
        self.inference.initialize(n_iter=T, n_print=10, n_samples=S)
        self.__run_inference__(T)

    def getTopWords(self, wordVec, tokens):
        K = self.K
        V = len(wordVec)
        qbeta = self.qbeta
        qbeta_sample = qbeta.eval()
        prob = [None] * K
        for k in range(K):
            prob[k] = qbeta_sample[:, k]
        self.tokens_probs = tokens_probs = [None] * K
        self.top_words = [None] * K
        for k in range(K):
            tokens_probs[k] = dict((t, p) for t, p in zip(range(V), prob[k]))
            newdict = sorted(tokens_probs[k],
                             key=tokens_probs[k].get,
                             reverse=True)[:15]
            self.top_words[k] = newdict
            print('topic %d' % k)
            for Id in newdict:
                print(tokens[Id], tokens_probs[k][Id])

    def getPMI(self, comatrix):
        K = self.K
        self.pmis = pmis = [None] * K
        for k in range(K):
            pmis[k] = util.pmi(comatrix, self.top_words[k])
            print('topic %d pmi: %f' % (k, pmis[k]))


class SimpleGaussianLDA(object):
    def __init__(self, K, D, N, nu, use_param=False):
        self.K = K  # number of topics
        self.D = D  # number of documents
        self.N = N  # number of words of each document
        self.nu = nu
        self.alpha = alpha = tf.zeros([K]) + 0.1
        self.sigmasq = InverseGamma(tf.ones(nu), tf.ones(nu), sample_shape=K)
        self.sigma = sigma = tf.sqrt(self.sigmasq)
        self.mu = mu = Normal(tf.zeros(nu), tf.ones(nu), sample_shape=K)
        self.theta = theta = [None] * D
        self.z = z = [None] * D
        self.w = w = [None] * D
        for d in range(D):
            theta[d] = Dirichlet(alpha)
            if use_param:
                w[d] = ParamMixture(
                    mixing_weights=theta[d],
                    component_params={'loc': mu, 'scale_diag': sigma},
                    component_dist=MultivariateNormalDiag,
                    sample_shape=N[d])
                z[d] = w[d].cat
            else:
                z[d] = Categorical(probs=theta[d], sample_shape=N[d])
                components = [
                    MultivariateNormalDiag(loc=tf.gather(mu, k),
                                           scale_diag=tf.gather(self.sigma, k),
                                           sample_shape=N[d])
                    for k in range(K)]
                w[d] = Mixture(cat=z[d],
                               components=components,
                               sample_shape=N[d])

    def __run_inference__(self, T, S=None):
        tf.global_variables_initializer().run()
        for n in range(self.inference.n_iter):
            info_dict = self.inference.update()
            self.inference.print_progress(info_dict)
        self.inference.finalize()

    def klqp(self, docs, S, T, wordVec):
        K = self.K
        D = self.D
        nu = self.nu
        self.latent_vars = latent_vars = {}
        training_data = {}
        qmu = Normal(
            loc=tf.Variable(tf.random_normal([K, nu])),
            scale=tf.nn.softplus(tf.Variable(tf.zeros([K, nu]))))
        latent_vars[self.mu] = qmu
        qsigmasq = InverseGamma(tf.nn.softplus(tf.Variable(tf.zeros([K, nu]))),
                                tf.nn.softplus(tf.Variable(tf.zeros([K, nu]))))
        latent_vars[self.sigmasq] = qsigmasq
        for d in range(D):
            training_data[self.w[d]] = docs[d]
        self.qmu = qmu
        self.qsigma = qsigma = tf.sqrt(qsigmasq)
        self.qw = MultivariateNormalDiag(loc=qmu, scale_diag=qsigma)
        V = len(wordVec)
        logprobs = [None] * V
        for i in range(V):
            logprobs[i] = self.qw.log_prob(wordVec[i])
        self.qbeta = tf.convert_to_tensor(logprobs)
        self.inference = ed.KLqp(latent_vars, data=training_data)
        self.inference.initialize(n_iter=T, n_print=10, n_samples=S)
        self.__run_inference__(T)

    def getTopWords(self, wordVec, tokens):
        K = self.K
        V = len(wordVec)
        qbeta = self.qbeta
        qbeta_sample = qbeta.eval()
        prob = [None] * K
        for k in range(K):
            prob[k] = qbeta_sample[:, k]
        self.tokens_probs = tokens_probs = [None] * K
        self.top_words = [None] * K
        for k in range(K):
            tokens_probs[k] = dict((t, p) for t, p in zip(range(V), prob[k]))
            newdict = sorted(tokens_probs[k],
                             key=tokens_probs[k].get,
                             reverse=True)[:15]
            self.top_words[k] = newdict
            print('topic %d' % k)
            for Id in newdict:
                print(tokens[Id], tokens_probs[k][Id])

    def getPMI(self, comatrix):
        K = self.K
        self.pmis = pmis = [None] * K
        for k in range(K):
            pmis[k] = util.pmi(comatrix, self.top_words[k])
            print('topic %d pmi: %f' % (k, pmis[k]))
