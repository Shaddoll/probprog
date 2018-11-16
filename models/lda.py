import tensorflow as tf
import edward as ed
import numpy as np
import pickle
from edward.models import Dirichlet, ParamMixture, Categorical, Empirical, \
                          WishartCholesky, MultivariateNormalTriL, \
                          Normal, Mixture
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
            print(d)
        self.inference = ed.Gibbs(latent_vars, proposal_vars, training_data)
        print("collapsed gibbs setup finished")
        self.inference.initialize(n_iter=T, n_print=1)
        print("initialize finished")
        self.__run_inference__(T)

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
        # self.ndk = ndk = [tf.Variable(_ndk[d]) for d in range(D)]
        self.ndk = ndk = tf.Variable(_ndk)
        # self.nvk = nvk = [tf.Variable(_nvk[v]) for v in range(V)]
        self.nvk = nvk = tf.Variable(_nvk)
        self.nk = nk = tf.reduce_sum(ndk, axis=0)
        # self.qz_prob = qz_prob = [(ndk[d] + alpha) * tf.gather(nvk, w[d])
        #                          / (nk + V * beta) for d in range(D)]
        # qz_prob = [qz_prob[d] / tf.reshape(tf.reduce_sum(
        #           qz_prob[d], axis=-1), (-1, 1)) for d in range(D)]
        qz_prob = [(ndk[d] + alpha) * tf.gather(nvk, w[d]) / (nk + V * beta) /
                   tf.reshape(tf.reduce_sum((ndk[d] + alpha) *
                                            tf.gather(nvk, w[d]) /
                                            (nk + V * beta), axis=-1),
                              (-1, 1)) for d in range(D)]
        # self.qz_prob = qz_prob = [[(ndk[d] + alpha) * nvk[wordIds[d][n]]
        #                           / (nk + V * beta) for n in range(N[d])]
        #                          for d in range(D)]
        self.qbeta = (tf.convert_to_tensor(nvk) + beta) / (nk + V * beta)
        self.update_ops = ops = [tf.no_op(), tf.no_op()]
        for d in range(D):
            print(d)
            for n in range(N[d]):
                with tf.control_dependencies([ops[-1], ops[-2]]):
                    oldk = qz[d][n]
                    # ops.append(tf.scatter_sub(ndk[d], [oldk], [1]))
                    ops.append(tf.scatter_nd_sub(ndk, [(d, oldk)], [1]))
                    # ops.append(tf.scatter_sub(nvk[w[d][n]], [oldk], [1]))
                    ops.append(tf.scatter_nd_sub(nvk, [(w[d][n], oldk)], [1]))
                # sequentially
                with tf.control_dependencies([ops[-1], ops[-2]]):
                    k = tf.multinomial(tf.log([qz_prob[d][n]]), 1,
                                       output_dtype=tf.int32)[0]
                    ops.append(tf.scatter_update(qz[d], [n], k))
                with tf.control_dependencies([ops[-1]]):
                    newk = qz[d][n]
                    # ops.append(tf.scatter_add(ndk[d], [newk], [1]))
                    ops.append(tf.scatter_nd_add(ndk, [(d, newk)], [1]))
                    # ops.append(tf.scatter_add(nvk[w[d][n]], [newk], [1]))
                    ops.append(tf.scatter_nd_add(nvk, [(w[d][n], newk)], [1]))
        sess = ed.get_session()
        tf.global_variables_initializer().run()
        # sess.run(tf.variables_initializer(qz + nvk + ndk))
        # print(sess.run(qz_prob))
        # print(sess.run(nvk))
        for _ in range(T):
            print("burnin:", _)
            sess.run(ops)
        qbeta_sample = np.zeros((V, K), dtype='float32')
        for _ in range(S):
            print("sample:", _)
            sess.run(ops)
            qbeta_sample += self.qbeta.eval()
        qbeta_sample /= S
        prob = [None] * K
        for k in range(K):
            prob[k] = qbeta_sample[:, k]
            print(len(prob[k]))
        tokens_probs = [None] * K
        for k in range(K):
            tokens_probs[k] = dict((t, p) for t, p in zip(tokens, prob[k]))
            newdict = sorted(tokens_probs[k],
                             key=tokens_probs[k].get,
                             reverse=True)[:15]
            print('topic %d' % k)
            for word in newdict:
                print(word, tokens_probs[k][word])

    def klqp(self, wordIds, S, T):
        K = self.K
        V = self.V
        D = self.D
        N = self.N
        latent_vars = {}
        training_data = {}
        self.qbeta_var = tf.nn.softplus(tf.Variable(tf.zeros([K, V]) + 1. / V),
                                        name="qbeta")
        qbeta = Dirichlet(self.qbeta_var)
        latent_vars[self.beta] = qbeta
        qtheta = [None] * D
        qz = [None] * D
        self.qtheta_var = [tf.nn.softplus(tf.Variable(tf.zeros([K]) + 1. / K),
                           name="qtheta%d" % d) for d in range(D)]
        self.qz_var = [tf.nn.softplus(tf.Variable(tf.zeros([K]) + 1. / K),
                       name="qz%d" % d) for d in range(D)]
        for d in range(D):
            qtheta[d] = Dirichlet(self.qtheta_var[d])
            latent_vars[self.theta[d]] = qtheta[d]
            qz[d] = Categorical(probs=self.qz_var[d], sample_shape=[N[d]])
            latent_vars[self.z[d]] = qz[d]
            training_data[self.w[d]] = wordIds[d]
        self.latent_vars = latent_vars
        self.inference = ed.KLqp(latent_vars, data=training_data)
        print("klqp setup finished")
        optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
        # optimizer = tf.train.GradientDescentOptimizer(0.001)
        self.inference.initialize(n_iter=T, n_print=1, n_samples=S,
                                  optimizer=optimizer, logdir='log/')
        self.__run_inference__(T, S)

    def criticize(self, tokens):
        K = self.K
        qbeta_sample = self.latent_vars[self.beta].sample().eval()
        pickle.dump(qbeta_sample, open("train2.p", "wb"))
        prob = [None] * K
        for k in range(K):
            prob[k] = qbeta_sample[k, :]
            print(len(prob[k]))
        tokens_probs = [None] * K
        for k in range(K):
            tokens_probs[k] = dict((t, p) for t, p in zip(tokens, prob[k]))
            newdict = sorted(tokens_probs[k],
                             key=tokens_probs[k].get,
                             reverse=True)[:15]
            print('topic %d' % k)
            for word in newdict:
                print(word, tokens_probs[k][word])

    # def evaluate(self, wordIds):
    #     latent_vars = {}
    #     latent_vars[self.beta] = self.qbeta
    #     for d in range(self.D):
    #         latent_vars[self.theta[d]] = self.qtheta[d]
    #         latent_vars[self.z[d]] = self.qz[d]
    #     self.latent_vars = latent_vars
    #     x_post = ed.copy(self.w, latent_vars)
    #     self.postlog = ed.evaluate('log_likelihood', data={x_post: wordIds})
    #     print (self.postlog)


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
        sigma_inv = tf.matrix_inverse(sigma)
        self.mu = mu = MultivariateNormalTriL(
            loc=mu0,
            scale_tril=sigma_inv)
        self.theta = theta = [None] * D
        self.z = z = [None] * D
        self.w = w = [None] * D
        for d in range(D):
            theta[d] = Dirichlet(alpha)
            if use_param:
                w[d] = ParamMixture(
                    mixing_weights=theta[d],
                    component_params={'loc': mu, 'scale_tril': sigma_inv},
                    component_dist=MultivariateNormalTriL,
                    sample_shape=N[d])
                z[d] = w[d].cat
            else:
                z[d] = Categorical(probs=theta[d], sample_shape=N[d])
                components = [
                    MultivariateNormalTriL(loc=tf.gather(mu, k),
                                           scale_tril=tf.gather(sigma_inv, k),
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

    def gibbs(self, docs, S, T):
        K = self.K
        D = self.D
        N = self.N
        nu = self.nu
        latent_vars = {}
        training_data = {}
        qmu = Empirical(tf.Variable(tf.zeros([S, K, nu])))
        latent_vars[self.mu] = qmu
        qsigma = Empirical(tf.Variable(tf.zeros([S, K, nu, nu])))
        latent_vars[self.sigma] = qsigma
        qtheta = [None] * D
        qz = [None] * D
        for d in range(D):
            qtheta[d] = Empirical(tf.Variable(tf.zeros([S, K]) + 0.1))
            latent_vars[self.theta[d]] = qtheta[d]
            qz[d] = Empirical(tf.Variable(tf.zeros([S, N[d]], dtype=tf.int32)))
            latent_vars[self.z[d]] = qz[d]
            training_data[self.w[d]] = docs[d]
        self.inference = ed.MetropolisHastings(latent_vars, data=training_data)
        print("gibbs setup finished")
        self.inference.initialize(n_iter=T, n_print=1)
        print("initialize finished")
        self.__run_inference__(T)

    def klqp(self, docs, S, T):
        K = self.K
        D = self.D
        nu = self.nu
        self.latent_vars = latent_vars = {}
        training_data = {}
        qmu = Normal(
            loc=tf.Variable(tf.random_normal([K, nu])),
            scale=tf.nn.softplus(tf.Variable(tf.zeros([K, nu]))))
        latent_vars[self.mu] = qmu
        qpsi0 = tf.Variable(tf.random_normal([K, nu, nu]))
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
        self.qpsi0=qpsi0
        self.Ltril=Ltril
        self.qmu=qmu
        self.qsigma=qsigma
        self.inference = ed.KLqp(latent_vars, data=training_data)
        self.inference.initialize(n_iter=T, n_print=10, n_samples=S)
        self.__run_inference__(T)
