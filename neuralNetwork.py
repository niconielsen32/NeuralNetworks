import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neural_network(inputs, targets, epochs, lr):
    
    W1 = np.random.rand(2,4)
    W2 = np.random.rand(4,1)
    
    for epoch in range(epochs):
        layer1 = sigmoid(np.dot(inputs, W1))
        output = sigmoid(np.dot(layer1, W2))
        error = targets - output
        delta2 = 2 * error * output * ( 1 - output)
        delta1 = delta2.dot(W2.T) * (layer1 * (1 - layer1))
        W2 += lr * layer1.T.dot(delta2)
        W1 += lr * inputs.T.dot(delta1)

    return np.round(output.squeeze())


inputs = np.array([[0,0],[0,1], [1,0], [1,1]])

print("OR", neural_network(inputs, np.array([[ 0.],[ 1.],[ 1.],[ 1.]]), 5000, 0.1))
print("AND", neural_network(inputs, np.array([[ 0.],[ 0.],[ 0.],[ 1.]]), 5000, 0.1))
print("XOR", neural_network(inputs, np.array([[ 0.],[ 1.],[ 1.],[ 0.]]), 5000, 0.1))
print("NAND", neural_network(inputs, np.array([[ 1.],[ 1.],[ 1.],[ 0.]]), 5000, 0.1))
print("NOR", neural_network(inputs, np.array([[ 1.],[ 0.],[ 0.],[ 0.]]), 5000, 0.1))
    