from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

for image in train_images:
    plt.imshow(image)
    plt.show()

