# one-layer network trained using simple delta rule

import numpy as np


class oln:
    def __init__(self, X, Y_target, learning_rate, epochs, linear=False):
        self.X = X
        self.Y_target = Y_target

        self.N = X.shape[0]
        self.d = X.shape[1]
        self.k = Y_target.shape[1]

        self.linear = linear

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.W = (1 / 280) * np.random.normal(
            0, 1, size=(self.k, self.d)
        )  # W initialization

        self.error_hist = np.zeros((epochs))

    def sigmoid(self, z):
        if self.linear == False:
            return 1 / (1 + np.exp(-2 * (z)))  # added bias
        else:
            return z

    def d_sigmoid(self, z):  # derivative of the sigmoid
        if self.linear == False:
            return (np.exp(-1 * z)) / (1 + np.exp(-1 * z)) ** 2
        else:
            return 1

    def learn_step(self):
        batch_size = 100  # take 100 random elements, s.t. not using 60,000 every epoch
        batch_idx = np.random.randint(self.N, size=batch_size)
        X_batch = self.X[batch_idx, :]
        Y_target_batch = self.Y_target[batch_idx, :]
        # X_batch = self.X[:batch_size, :]
        # Y_target_batch = self.Y_target[:batch_size, :]

        delta_W = np.zeros(
            self.W.shape
        )  # accumulate all errors from this batch and update at the end

        batch_error = 0  # total error across the whole batch

        # writing this to maximize readability, at this point
        for i in range(batch_size):
            x = X_batch[i]
            #x = np.insert(x, 0, 1, axis=0)  # adding bias term
            y_target = Y_target_batch[i]

            u = self.W @ x  # temporary variable
            y = self.sigmoid(u)

            e = (1 / 2) * np.dot((y - y_target).T, (y - y_target))  # mean-squared error
            batch_error += e / batch_size

            de = self.d_sigmoid(u) * (
                y_target - y
            )  # derivative of the error w respect to W
            # de = y - y_target
            delta_W = np.outer(de, x)
            self.W += self.learning_rate * delta_W

        return batch_error

    def run(self):
        for i in range(self.epochs):
            if (i % 1000 == 0):
                print("training epoch {}".format(i))
            e = self.learn_step()
            self.error_hist[i] = e

        return

    def encode_input(self, X):
        Y = np.zeros((X.shape))

        for i in range(X.shape[0]):
            # x = np.insert(X[i], 0, 1, axis=0)  # adding bias term
            x = X[i]
            u = self.W @ x
            Y[i] = self.sigmoid(u)

        return Y
