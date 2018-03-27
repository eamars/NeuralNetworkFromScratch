from src.loader import load_data
import matplotlib.pyplot as plt

from src.network0 import Network

def print_training_data(item):
    image = item[0]
    label = item[1]

    plt.imshow(image.reshape(28, 28))
    plt.show()

training_data, testing_data = load_data()

print_training_data(training_data[0])

net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, testing_data)