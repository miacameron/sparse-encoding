
import numpy as np
import gzip
import pickle

from rnn import rnn
from threshold import threshold_sparsity
from foldiak import anti_hebbian

url = "./ah_1000/"

print("testing pickle")

a = 10
with open(url + 'test.pkl', 'wb') as f:
    pickle.dump(a, f)


print("begin")
#importing mnist
src = "../data/train-images-idx3-ubyte.gz"
with gzip.open(src, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
X = data.reshape(data.shape[0], 28*28).astype(np.float64)

# Min-max normalizing
for t in range(X.shape[0]):
    if (np.linalg.norm(X[t]) != 0):
        X[t] = (X[t] - np.min(X[t])) / (np.max(X[t]) - np.min(X[t]))
print("imported mnist")

print("training thresholds")

antihebbian = anti_hebbian(784, 784)
antihebbian.train_thresholds(X[:1000], 20)
print("finished training thresholds")
antihebbian.train(X[:1000], 1000)

antihebbian_learned_params = [antihebbian.W, antihebbian.Q, antihebbian.t]

with open(url + 'antihebbian_learned_params_1000.pkl', 'wb') as f:
    pickle.dump(antihebbian_learned_params, f)

