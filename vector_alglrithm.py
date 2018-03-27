import numpy as np
import matplotlib.pyplot as plt

def native_relu(x):
    assert(len(x.shape) == 2)

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def native_add(x, y):
    assert(len(x.shape[0]) == 2)
    assert(x.shape == y.shape)

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x


def native_add_matrix_and_vector(x, y):
    assert(len(x.shape) == 2) # 2d matrix
    assert(len(y.shape) == 1) # 1d array
    assert(x.shape[1] == y.shape[0])

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i]

    return x


def native_vector_dot(x, y):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    assert(x.shape == y.shape) # make sure x and y have same length

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]

    return z


def native_matrix_vector_dot(x, y):
    assert(len(x.shape) == 2) # matrix
    assert(len(y.shape) == 1) # vector
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0]) # the result vector matches the rows of x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]

    return z

def native_matrix_vector_dot_2(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = native_vector_dot(x[i, :], y)
    return z


def native_matrix_dot(x, y):
    assert(len(x.shape) == 2) # 2d matrix
    assert(len(y.shape) == 2) # 2d matrix

    assert(x.shape[1] == y.shape[0])

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row = x[i, :] # grab the row of x
            col = y[:, j] # and grab the col of y, then find the dot product of two
            z[i, j] = native_vector_dot(row, col)
    return z


x = np.random.random((3, 3))
print(x)
w = x.reshape((1, 9))
print(w)
v = w.transpose()
print(v)
