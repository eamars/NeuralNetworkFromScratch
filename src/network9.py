"""
Homebrew neural network core
"""

import numpy as np
import random


def sigmoid(z):
    # sigmoid function that maps data to points between 0 to 1
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # dt of sigmoid
    return sigmoid(z) * (1-sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # number of neurons in the respective layer

        # generate Gaussian distribution with mean 0 and standard deviation of 1
        # the first layer is the input layer so we skip the bias for it
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # a' = sig(wa + b)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def df_cost(self, activation, label):
        return activation - label

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]
        return sum(int(y[x] == 1) for x, y in test_results)

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def back_propagation(self, data, label):
        # set activation x
        a = data

        # create list to store activations and z values
        activation_list = [a]
        z_list = []

        # perform feed forward from layer 2 to layer L
        for weight, bias in zip(self.weights, self.biases):
            # calculate new z with weight^l * a^(l-1) + b^l
            z = np.dot(weight, a) + bias
            z_list.append(z)

            # update new activation with sigmoid function
            a = sigmoid(z)
            activation_list.append(a)

        # calculate output errors at output layer
        # error^L = dfC * sigmoid'(z^L) where
        # cost = 1/2 * (y - a)^2, the derivative of cost will be
        # cost' = (y - a)
        error = self.df_cost(activation_list[-1], label) * sigmoid_prime(z_list[-1])

        # we can combine the back propagation and gradient descent together
        # for better efficiency
        df_bias = [np.zeros(bias.shape) for bias in self.biases]
        df_weight = [np.zeros(weight.shape) for weight in self.weights]

        # calculate partial derivative of bias and weight according to
        # BP3 and BP4
        df_bias[-1] = error
        df_weight[-1] = np.dot(error, activation_list[-1 - 1].transpose())

        # perform gradient descent with above algorithm
        # note: the layer here doesn't truly represent the neural network
        # layer. When layer = 1 means the last layer of neurons, layer = 2
        # is the second last layer (in this case we utilize the python's
        # negative indices
        for layer in range(2, self.num_layers):
            error = np.dot(self.weights[-layer + 1].transpose(), error) * sigmoid_prime(z_list[-layer])
            df_bias[-layer] = error
            df_weight[-layer] = np.dot(error, activation_list[-layer - 1].transpose())

        return df_bias, df_weight

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # define number of tests with test_data input
        num_tests = 0
        if test_data is not None:
            num_tests = len(test_data)

        num_training_data = len(training_data)

        # depending on epochs (numbers of repeated training), we gonna execute each
        # iteration with different order of inputs
        for iterations in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_training_data, mini_batch_size)]

            for mini_batch in mini_batches:
                df_bias = [np.zeros(bias.shape) for bias in self.biases]
                df_weight = [np.zeros(weight.shape) for weight in self.weights]

                # for each training sample, set training activations
                for data, label in mini_batch:
                    # accumulate delta gradient descent
                    delta_df_bias, delta_df_weight= self.back_propagation(data, label)
                    df_bias = [nb + dnb for nb, dnb in zip(df_bias, delta_df_bias)]
                    df_weight = [nw + dnw for nw, dnw in zip(df_weight, delta_df_weight)]

                # update bias and weights
                self.weights = [w - (eta / len(mini_batch) * wd) for w, wd in zip(self.weights, df_weight)]
                self.biases = [b - (eta/len(mini_batch) * bd) for b, bd in zip(self.biases, df_bias)]

            # done one iteration, we want to evaluate the performance of the network
            if test_data:
                print("Epoch {0}: {1}/{2}".format(iterations, self.evaluate(test_data), num_tests))
            else :
                print("Epoch {0} complete")
