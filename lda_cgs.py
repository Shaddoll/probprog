import tensorflow as tf
import edward as ed
import glob
from edward.models import Dirichlet, ParamMixture, Categorical, Empirical
import numpy as np


class CollapsedGibbsLDA(object):
    def __init__(self, path_data, n_topics):
        self.n_topics = n_topics     # number of topics
        self.corpus = {}          # TRAINING DATA: {DocID: [WordID1, WordID1, WordID2, ...]}
        self.doc_to_id = {}          # MAP DOCUMENT INTO INDEX: self.indD = {DocID: INDEX}
        self.vocab_to_id = {}        # MAP WORD INTO INDEX: self.indV = {VocabID: INDEX}
        self.id_to_vocab = {}
        self.n_docs = 0                # NUMBER OF DOCUMENTS
        self.n_vocab = 0              # NUMBER OF VOCABULARIES
        self.nwords_per_docs = [0] * self.n_docs # number of words of each document

        self.alpha = tf.zeros([self.n_topics])    #prior for theta, d x k
        self.yita =  tf.zeros([self.n_vocab]) + 0.05
        self.theta = [None] * self.n_docs
        self.beta = Dirichlet(self.yita, sample_shape=n_topics)
        self.z = [None] * self.n_docs
        self.w = [None] * self.n_docs
        for d in range(self.n_docs):
            self.theta[d] = Dirichlet(self.alpha)
            self.w[d] = ParamMixture(mixing_weights=self.theta[d],
                                     component_params = {'probs': self.beta},
                                     component_dist = Categorical,
                                     sample_shape = self.nwords_per_docs[d],
                                     validate_args=True)
            self.z[d] = self.w[d].cat



    def load_data(self, datapath,vocabpath):
        txt_files = glob.glob(datapath)
        self.n_docs = len(txt_files)
        print("number of documents, D: {}".format(self.n_docs))
        count = 0
        for file in (txt_files):
            self.doc_to_id[file[-8:-4]] = count
            with open(file, 'rt', encoding="ISO-8859-1") as f:
                self.corpus[file[-8:-4]] = list(map(int, f.readline().split()))
                self.nwords_per_docs[count] = len(self.corpus[count])
                self.corpus[count] = np.array(self.corpus[count]).astype('int32')
            print("load" + file + "finished")
            count += 1
        with open(vocabpath) as f:
            for line in f:
                self.n_vocab += 1
                line = line.split()
                self.vocab_to_id[line[0]] = int(line[1])
                self.id_to_vocab[int(line[1])] = line[0]
        self.tokens = self.vocab_to_id.keys()
        print("vocab size is {}".format(self.n_vocab))
        print("load wordToIDtoy.txt finished")

    def assignTopics(self, doc, token, pos_j):  # DROW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
        d_i = self.doc_to_id[doc]
        w_ij = self.vocab_to_id[token]
        z_ij = self.topicAssignments[d_i][pos_j]  # TOPIC ASSIGNMENT OF WORDS FOR EACH DOCUMENT
        self.cnt_topic_word[z_ij, w_ij] -= 1     # NUMBER OF TOPICS ASSIGNED TO A WORD
        self.cnt_doc_topic[d_i, z_ij] -= 1     # NUMBER OF TOPICS ASSIGNED IN A DOCUMENT
        self.cnt_topic[z_ij] -= 1
        self.lenD[d_i] -= 1

        # FULL-CONDITIONAL DISTRIBUTION
        prob_topic_doc = (self.cnt_doc_topic[d_i] + self.alpha) / (self.lenD[d_i] + tf.reduce_sum(self.alpha))
        ### probability of token observed in topic
        prob_token_topic = (self.cnt_topic_word[:, w_ij] + self.yita) / (self.cnt_topic + self.yita * self.n_vocab)
        prFullCond = prob_topic_doc * prob_token_topic  # FULL-CONDITIONAL DISTRIBUTION
        prFullCond /= tf.reduce_sum(prFullCond)  # TO OBTAIN PROBABILITY
        # NOTE: 'prFullCond' is MULTINOMIAL DISTRIBUTION WITH THE LENGTH, NUMBER OF TOPICS, NOT A VALUE
        new_z = tf.multinomial(1, prFullCond).argmax()  # RANDOM SAMPLING FROM FULL-CONDITIONAL DISTRIBUTION
        self.topicAssignments[d_i][pos_j] = new_z
        self.cnt_topic_word[z_ij, w_ij] += 1
        self.cnt_doc_topic[d_i, z_ij] += 1
        self.cnt_topic[z_ij] += 1
        self.lenD[d_i] += 1

    def LogLikelihood(self):  # FIND (JOINT) LOG-LIKELIHOOD VALUE
        l = 0
        for z in range(self.n_topics):  # log p(w|z,\beta)
            l += gammaln(self.n_vocab * self.beta)
            l -= self.n_vocab * gammaln(self.beta)
            l += np.sum(gammaln(self.cntTW[z] + self.beta))
            l -= gammaln(np.sum(self.cntTW[z] + self.beta))
        for doc in self.documents:  # log p(z|\alpha)
            d = self.indD[doc]
            l += gammaln(np.sum(self.alpha))
            l -= np.sum(gammaln(self.alpha))
            l += np.sum(gammaln(self.cntDT[d] + self.alpha))
            l -= gammaln(np.sum(self.cntDT[d] + self.alpha))
        return l

    def find_alpha_yita(self):
        # ADJUST ALPHA AND YITA BY USING MINKA'S FIXED-POINT ITERATION
        numerator = 0
        denominator = 0
        for d in range(self.n_docs):
            numerator += tf.digamma(self.cnt_doc_topic[d] + self.alpha) - tf.digamma(self.alpha)
            denominator += tf.digamma(tf.reduce_sum(self.cnt_doc_topic[d] + self.alpha)) - tf.digamma(tf.reduce_sum(self.alpha))
        self.alpha *= numerator / denominator  # UPDATE ALPHA
        numerator = 0
        denominator = 0
        for z in range(self.n_topics):
            numerator += tf.reduce_sum(tf.digamma(self.cnt_topic_word[z] + self.yita) - tf.digamma(self.yita))
            denominator += tf.digamma(tf.reduce_sum(self.cnt_topic_word[z] + self.yita)) - tf.digamma(self.n_vocab * self.yita)
        self.yita = (self.yita * numerator) / (self.n_vocab * denominator)  # UPDATE YITA

    def find_theta_beta(self):
        theta_update = tf.zeros((self.n_docs, self.n_topics))  # SPACE FOR THETA
        beta_update = tf.zeros((self.n_topics, self.n_vocab))  # SPACE FOR BETA
        for d in range(self.n_docs):
            for z in range(self.n_topics):
                theta_update[d][z] = (self.cnt_doc_topic[d][z] + self.alpha[z]) / (self.lenD[d] + tf.reduce_sum(self.alpha))
        for z in range(self.n_topics):
            for w in range(self.n_vocab):
                beta_update[z][w] = (self.cnt_topic_word[z][w] + self.yita) / (self.cnt_topic_word[z] + self.beta * self.n_vocab)
        return beta_update, theta_update

    def run(self, nsamples, burnin, interval):  # GIBBS SAMPLER KERNEL
        if nsamples <= burnin:  # BURNIN CHECK
            print("ERROR: BURN-IN POINT EXCEEDS THE NUMBER OF SAMPLES")
            sys.exit(0)
        print("# of DOCS:", self.n_docs)  # PRINT TRAINING DATA INFORMATION
        print("# of TOPICS:", self.n_topics)
        print("# of VOCABS:", self.n_vocab)

        # MAKE SPACE FOR TOPIC-ASSIGNMENT MATRICES WITH 0s
        self.topicAssignments = {}  # {INDEX OF DOC: [TOPIC ASSIGNMENT]}
        for doc in self.corpus:
            d = self.doc_to_id[doc]
            self.topicAssignments[d] = [0 for word in self.corpus[doc]]
        self.cnt_topic_word = tf.zeros((self.n_topics, self.n_vocab))  # NUMBER OF TOPICS ASSIGNED TO A WORD
        self.cnt_doc_topic = tf.zeros((self.n_docs, self.n_topics))  # NUMBER OF TOPICS ASSIGNED IN A DOCUMENT
        self.cnt_topic = tf.zeros(self.n_topics)  # ASSIGNMENT COUNT FOR EACH TOPIC
        self.lenD = tf.zeros(self.n_docs)  # ASSIGNMENT COUNT FOR EACH DOCUMENT = LENGTH OF DOCUMENT

        # RANDOMLY ASSIGN TOPIC TO EACH WORD
        for doc in self.corpus:
            for i, word in enumerate(self.corpus[doc]):
                d = self.doc_to_id[doc]
                w = self.vocab_to_id[word]
                random_topic = tf.random_uniform(shape=(), minval=0,
                                                 maxval=self.n_topics - 1)  # RANDOM TOPIC ASSIGNMENT
                self.topicAssignments[d][i] = random_topic  # RANDOMLY ASSIGN TOPIC TO EACH WORD
                self.cnt_topic_word[random_topic, w] += 1
                self.cnt_doc_topic[d, random_topic] += 1
                self.cnt_topic[random_topic] += 1
                self.lenD[d] += 1

        # COLLAPSED GIBBS SAMPLING
        print("INITIAL STATE")
        print("\tLikelihood:", self.LogLikelihood())  # FIND (JOINT) LOG-LIKELIHOOD
        print("\tAlpha:", end="")
        for i in range(self.n_topics):
            print(" %.5f" % self.alpha[i], end="")
        print("\n\tBeta: %.5f" % self.beta)
        SAMPLES = 0
        for s in range(nsamples):
            for doc in self.corpus:
                for i, word in enumerate(self.corpus[doc]):
                    self.assignTopics(doc, word, i)  # DROW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
            self.find_alpha_yita()  # UPDATE ALPHA AND YITA VALUES
            lik = self.LogLikelihood()
            print("SAMPLE #" + str(s))
            print("\tLikelihood:", lik)
            print("\tAlpha:", end="")
            for i in range(self.n_topics):
                print(" %.5f" % self.alpha[i], end="")
            print("\n\tYita: %.5f" % self.yita)
            if s > burnin and s % interval == 0:  # FIND PHI AND THETA AFTER BURN-IN POINT
                beta_update, theta_update = self.find_theta_beta()
                self.theta += theta_update
                self.beta += beta_update
                SAMPLES += 1
        self.theta /= SAMPLES  # AVERAGING GIBBS SAMPLES OF THETA
        self.beta /= SAMPLES  # AVERAGING GIBBS SAMPLES OF BETA
        return lik