import numpy as np

class rnn:

    def __init__(self, X, learning_rate, epochs, linear = False):

        self.X = X
        self.T = X.shape[0]
        self.d = X.shape[1]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.linear = linear

        self.R = 0.001 * np.random.normal(0, 1, size=(self.d, self.d + 1))

        self.error_hist = np.zeros((epochs))

    def sigmoid(self, z):
        if (self.linear == False):
            return 1 / (1 + np.exp( -2 * (z))) # added bias
        else:
            return z

    def d_sigmoid(self, z): # derivative of the sigmoid
        if (self.linear == False):
            return (np.exp( -1 * z)) / (1 + np.exp( -1 * z))**2
        else:
            return 1
        
    def learn_step (self):

        delta_R = np.zeros(self.R.shape) # accumulate all errors from this sequence and update at the end
        error = 0 # total error across the whole sequence

        Y = np.zeros((self.X.shape))

        # writing this to maximize readability, at this point
        for i in range(self.T - 1):

            x_prev = self.X[i]
            x_next = self.X[i + 1]
            x_prev = np.insert(x_prev, 0, 1, axis=0) #adding bias term

            u = self.R @ x_prev # temporary variable
            y_next = self.sigmoid(u)

            e = (1/2) * np.dot((y_next - x_next).T, (y_next - x_next)) # mean-squared error
            error += e

            de = self.d_sigmoid(u) * (x_next - y_next) # derivative of the error respect to R
            delta_R = np.outer(de, x_prev)
            self.R += self.learning_rate * delta_R
        
        return error
    

    def run(self):

        for i in range(self.epochs):
            self.error_hist[i] = self.learn_step()
        
        return

    def recall(self, X_0, T):

        X_rec = np.zeros((T, X_0.shape[0]))
        X_rec[0] = X_0

        for i in range(1,T):
            X_rec[i] = self.sigmoid( self.R @ np.insert(X_rec[i-1], 0, 1, axis=0) )
        
        return X_rec