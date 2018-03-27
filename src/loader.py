import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    training_inputs = [image.reshape(28*28, 1).astype('float32') / 255 for image in train_images]
    training_results = [vectorized_result(label) for label in train_labels]

    training_data = list(zip(training_inputs, training_results))

    testing_inputs = [image.reshape(28*28, 1).astype('float32') / 255 for image in test_images]
    testing_results = [vectorized_result(label) for label in test_labels]

    testing_data = list(zip(testing_inputs, testing_results))

    return training_data, testing_data
