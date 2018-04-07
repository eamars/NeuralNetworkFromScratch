import numpy as np
import random


def sigmoid(z):
    # sigmoid function that maps data to points between 0 to 1
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # dt of sigmoid
    return sigmoid(z) * (1-sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # number of neurons in the respective layer

        # generate Gaussian distribution with mean 0 and standard deviation of 1
        # the first layer is the input layer so we skip the bias for it
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # a' = sig(wa + b)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        :param training_data: (training input, desired output)
        :param epochs: numbers of epoches to train for
        :param mini_batch_size: the size of mini_batch when sampling data
        :param eta: the learning rate
        :param test_data:
        :return:
        """
        n_test = 0

        if test_data is not None:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                # part the training data into smaller mini batches
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                # apply a single step of gradient descent
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nb + dnw for nb, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # compute the gradient of the cost function
        # why do we need x and y in this case?
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward
        activation = x
        activations = [x] # lists to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(y[x] == 1) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


