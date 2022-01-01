import numpy as np


# calculates value of sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# calculates value of derivative of sigmoid function
def sigmoid_der(x):
    return x * (1 - x)  # TODO funkcja straty dla [0, 1]
    #s = 1 / (1 + np.exp(-x))
    #return s * (1 - s)


def loss(y_predicted, y_correct):
    return (y_predicted - y_correct) ** 2


def loss_der(y_predicted, y_correct):
    return 2 * (y_predicted - y_correct)
