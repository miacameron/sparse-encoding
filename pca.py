import numpy as np
from numpy.linalg import eig

class pca:

    def __init__(self, dim):
        self.d = dim
        return
    
    def encode_inputs(self, X):
        N = X.shape[0] # num. data points

        x_mean = (1/N) * np.sum(X, axis=0)
        X_0 = X - x_mean # deviation from the mean

        C = (1/(N - 1)) * np.matmul(X_0.T, X_0) # computing the covariance matrix
        lam, V = eig(C) # eigenvalues and right eigenvectors
        U = V[:self.d,:] # d eigenvalues with d-largest eigenvalues

        Y = np.zeros((N,self.d)) # matrix of transformed data points in 
        for i in range(N):
            Y[i,:] = np.dot(U, X_0[i,:])

        return Y