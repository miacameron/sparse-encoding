import numpy as np


def gaussian_activation(t, event_t, interval, sigma, peak=0.1):

    return peak * np.exp(-((t - event_t + interval/2) % interval - interval/2) ** 2 / (2 * sigma**2)) / (np.sqrt(2*np.pi)*sigma)


def create_inputs(n_inputs, sim_time, interval, sigma=0.5):

    ts = np.arange(0, sim_time).astype("float32")
    peak = sigma*2
    # Events are distributed "diagonally"
    events = np.arange(1, interval + 1, interval/n_inputs)
    ec_inputs = np.zeros((sim_time, n_inputs))

    for i in range(0, n_inputs):
        ec_inputs[:, i] = gaussian_activation(
            ts, events[i], interval, sigma, peak)

    # normalize ec_inputs across neurons at each timestep, i.e. make they sum to 1, so that they can be represented using a softmax function
    for t in range(sim_time):
        ec_inputs[t,:] = ec_inputs[t,:] / np.linalg.norm(ec_inputs[t,:])

    return ec_inputs


def shuffle_inputs(inputs):
    T, N = inputs.shape
    rand_idx = np.random.permutation(N)
    return np.array(inputs[:, rand_idx]), rand_idx


def add_noise(inputs, noise=0.01, sd=0.3):

    noisy_input = (1.0-noise)*inputs + noise * \
        np.random.normal(np.mean(inputs), scale=sd, size=(inputs.shape))
    return noisy_input


def sparsify_input(input, sparsity):
    IO_ratio = int(1 / sparsity)
    N_sparse = input.shape[1] * IO_ratio

    output = np.zeros((1, N_sparse))

    for i in range(0, input.shape[0]):
        output[:, i*IO_ratio] = input[:, i]
    return output


def generate_mask(N, p):
    A = np.random.choice([0, 1], size=(N, N), p=[1-p, p])
    return A



def generate_lines_input(N):

    X = []

    for _ in range(N):

        x = np.zeros((8,8))

        for i in range(16):
            if (np.random.randint(0,8) == 0):
                if (i < 8):
                    x[i,:] = 1
                else:
                    x[:,(i-8)] = 1
        
        X.append(x)
    
    return np.array(X)
