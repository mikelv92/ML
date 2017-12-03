from PIL import Image
import numpy as np
import cPickle as pickle

H = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
resume = False

D = 256 * 240 * 3

def get_parameters(layer_dims, resume):
    if resume:
        parameters = pickle.load('savedModel.p', 'rb')
    else:
        parameters = {}
        L = len(layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(2 / (layer_dims[l] + layer_dims[l - 1]))
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z)), z

def relu(z)
    return z * (z > 0), z

def linear_forward(A, W, b):
    Z = np.dot(A, W) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    
    cache = (linear_cache, activation_cache)
    
    return A, cache

def forward(x, parameters):
    caches = []
    A = x
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches 

def compute_cost(AL, y):
    pass



def prepro(I):
    pass

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0

    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0
        runnig_add = running_add + gamma * r[t]
        discounted_r[t] = running_add
    
    return discounted_r

def policy_forward(model, x):

    z_cache = []
    a_cache = []

    h1 = np.dot(model['W1'], x)
    h1_relu = relu(h1)
    h2 = np.dot(model['W2'], h1)
    h2_sigmoid = sigmoid(h2)

    z_cache.append(h1)
    z_cache.append(h2)

    a_cache.append(h1_relu)
    a_cache.append(h2_sigmoid)

    return a_cache, z_cache
    
def policy_backward(hs, dALs):
    

if __name__ == '__main__':

    timestamp = 0
    last_timestamp = 0
    file_ready = False
    mario_position = []
    enemies = []

    while True:
        with open("extracted_data.txt", "r") as f:
            lines = f.readlines()

            enemies = []

            for i in range(0, len(lines)):
                line = lines[i].strip()
                tokens = line.split(":")

                if tokens[0] == 'MarioPosition':
                    mario_position = tokens[1].split(",")
                elif tokens[0].startswith('Enemy'):
                    enemies.append(tokens[1].split(","))
                elif tokens[0] == 'Timestamp':
                    timestamp = tokens[1]
                    file_ready = True

        # Timestamp is the last line in the file. If it is not found, the data is not complete in the file. 
        # If the timestamp is the same as the old one, there is no new data in the file.

        if not file_ready or timestamp == last_timestamp:
            continue

        im = Image.open("screenshot.png")

        with open("commands.txt", "w+") as f:
            f.write("true\n")
            f.write("false\n")
            f.write("false\n")
            f.write("false\n")
            f.write("true\n")
            f.write("false\n")

        last_timestamp = timestamp
        file_ready = False
        sleep(0.05)
