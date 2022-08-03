import numpy as np
import pandas as pd

data = pd.read_csv(r"mnist_train.csv")
data = np.array(data)
(m, n) = data.shape

# m = 785
# n = 60000
np.random.shuffle(data)
# Transpose the array, now each colum represents a "number" with its pixel data
test_data = data[0:1000].transpose()
test_y = test_data[0]
test_x = test_data[1:n]
# To evaluate overfitting, the data set is split into testing and training subsets
train_data = data[1000:m].transpose()
train_y = train_data[0]
train_x = train_data[1:n]
train_x = (train_x / 255)


def init_params():
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2


def ReLU(Z):
    # if Z is grater than 1, return that value. If it is not, return 0. Used to simulate the "all or nothing" phenomena
    # of a neuron's action potential
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward_propagation(W1, b1, W2, b2, x):
    # Z1 is the unactivated, hidden layer
    Z1 = W1.dot(x) + b1
    # A1 is the activated layer
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    # sets the labels into a list with 10 zeroes indicating the number. Then we take the actual number through an
    # index and set that indexed zero to one [0 0 0 0 1 0 0 0 0 0] would be the label for the number 4
    one_hot_Y = np.zeros((Y.size, 10))
    # Shape of zero array would be y.size rows (total number of examples) and the length of each row would be the
    # y.max (9) + 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Change the index value at each row to 1 indicating the label
    one_hot_Y = one_hot_Y.transpose()
    # transpose so each column represents a number rather than row

    return one_hot_Y


def deriv_ReLU(x):
    return x > 0


def backwards_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = (A2 - one_hot_Y)
    dW2 = 1 / m * dZ2.dot(A1.transpose())
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.transpose().dot(dZ2) * deriv_ReLU(Z2)
    dW1 = 1 / m * dZ1.dot(X.transpose())
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dW2, dB1, dB2


def update_param(W1, B1, W2, B2, dW1, dW2, dB1, dB2, alpha):
    # After going through forward prop and backwards prop, multiply the values returned from back prop (amount of change
    # needed for each weight and bias) by the learning rate alpha, then subtracting from original weights and biases
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, W2, B1, B2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def NN(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, dW2, dB1, dB2 = backwards_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, W2, b1, b2 = update_param(W1, b1, W2, b2, dW1, dW2, dB1, dB2, alpha)
        # Every 50 iterations, give an update on the relative accuracy of the network
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = NN(train_x, train_y, 4000, 0.1)
# Save the arrays of the weights and biases of each layer
np.savetxt('W1.out', W1, delimiter=',')
np.savetxt('b1.out', b1, delimiter=',')
np.savetxt('W2.out', W2, delimiter=',')
np.savetxt('b2.out', b2, delimiter=',')
