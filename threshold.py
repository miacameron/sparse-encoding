import numpy as np

class threshold_sparsity:

    def __init__(self, threshold):

        self.threshold = threshold

    def encode_inputs(self, X):
        
        N = X.shape[0] # num. data points
        d = X.shape[1]

        k = int( self.threshold * d ) # num. units to keep active
        print(k)

        for i in range(N):
            inx = np.argpartition(X[i], (d - k))[:(d - k)]
            X[i][inx] = 0

        return X
