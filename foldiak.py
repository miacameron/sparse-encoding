import numpy as np


class anti_hebbian:

    def __init__(self, n, m):

        self.n = n # representation units
        self.m = m # input units

        self.Q = np.random.uniform(0,1, (self.n, self.m)) # input -> representation weights
        row_sum = np.linalg.norm(self.Q, axis=-1) # normalizing
        self.Q = self.Q / row_sum[:,np.newaxis]

        self.W = np.zeros((self.n, self.n)) # representation reccurrent weights

        # learning parameters
        self.alpha = 0.1
        self.beta = 0.02
        self.gamma = 0.02
        self.lam = 10.0
        self.p = 0.05 # approximately 5% sparsity

        self.tau = 0.02
        self.cyc = 100

        self.y_thr = 0.5

        self.y = np.zeros((self.n)) # current state of representation units
        self.t = np.zeros((self.n)) # threshold

        self.dQ_hist = []
        self.dW_hist = []
        self.dt_hist = []
    
    def f(self, u):
        return 1/ (1 + np.exp( - self.lam * u))

    def run(self, cycles : int, x):

        self.y = np.zeros((self.n))
        #dy = self.f( (self.Q @ x) + (self.W @ self.y) - self.t) - self.y
        #self.y += dy
        
        for i in range(cycles):
            #print("cycle {}".format(i))
            Qx = self.Q @ x
            Wy = self.W @ self.y
            dy = self.f( Qx.T + Wy - self.t) - self.y
            self.y += self.tau * dy
        #print(self.y)

        self.y = np.array([1.0 if y_ > self.y_thr else 0.0 for y_ in self.y])
        return self.y
        
    def train_step(self, x):

        #anti-hebbian 
        dw = -self.alpha * (np.outer(self.y, self.y) - self.p**2)
        self.W += dw
        np.fill_diagonal(self.W, 0)
        self.W = self.W.clip(max=0)
        #dW = np.zeros((self.W.shape))
        #for i in range(self.W.shape[0]):
        #    for j in range(self.W.shape[1]):
        #        if (i==j):
        #            pass
        #        dW[i,j] = - self.alpha * (self.y[i] * self.y[j] - self.p**2)
        #self.W = self.W.clip(max=0)
        #np.fill_diagonal(self.W, 0)
        #self.W += dW
        self.dW_hist.append(np.sum(np.sum(abs(dw))))

        if(not np.allclose(self.W.T, self.W)):
            print("W not symmetric")
            return
        #print(np.sum(dw))

        # hebbian
        dq = self.beta * (np.outer(self.y, x) - (self.Q * self.y[:,np.newaxis]))
        self.Q += dq
        #print(np.sum(dq))
        #dQ = np.zeros((self.Q.shape)) # TODO vectorize this later
        #for i in range(self.Q.shape[0]):
        #    for j in range(self.Q.shape[1]):
        #        dQ[i,j] = self.beta * self.y[i] * (x[j] - self.Q[i,j])
        
        #self.Q += dQ
        self.dQ_hist.append(np.sum(np.sum(abs(dq))))

        # threshold modification
        dt = self.gamma * (self.y - self.p)
        self.t += dt
        self.dt_hist.append(np.sum(np.sum(abs(dt))))
        #print(dt)


    def train(self, X, epochs):
        self.dQ_hist = []
        self.dW_hist = []
        self.dt_hist = []
        
        for e in range(epochs):
            if (e % 100 == 0):
                print("training epoch {}".format(e))
            for i in range(X.shape[0]):
                self.run(self.cyc, X[i])
                self.train_step(X[i])
    

    def train_thresholds(self, X, epochs):
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.01
        self.train(X, epochs)

        self.alpha = 0.1
        self.beta = 0.02
        self.gamma = 0.02
        return
    

    def encode_inputs(self, X):
        Y = np.zeros((X.shape[0], self.n))

        for i in range(X.shape[0]):
            Y[i] = self.run(self.cyc, X[i])

        return Y