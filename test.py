from models.lda import GaussianLDA

N = [10, 11, 20, 30]
D = len(N)
K = 5
nu = 50

model = GaussianLDA(K, D, N, nu)
