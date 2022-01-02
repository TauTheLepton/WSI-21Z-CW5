from datamanager import *
from aimathfunctions import *


# class doing all neural network calculations and storing its data
class NeuralNetwork:
    """
    learn_data: array-like with data to learn from, size of learn data decide about number of input neurons
    hidden_num: number of neurons in hidden layer of neural network // change to hidden_layer_size?
    output_num: number of neurons in output layer of neural network //tak samo jak wyżej?
    """
    def __init__(self, learn_data, hidden_num, output_num):
        self.inputs = self.setup_inputs(learn_data)
        self.hidden = []
        self.outputs, self.outputs_correct = self.setup_outputs(learn_data)
        self.weights1, self.weights2 = self.setup_weights(hidden_num, output_num)
        self.errors = []
        self.values = []

    def setup_inputs(self, data): # TODO zamienić na dekorator
        d_idx = getDIdx(data)
        inputs = normalize(data[:, :d_idx])
        return inputs
        #bias = np.ones((len(inputs), 1))
        #return np.append(inputs, bias, 1)

    def setup_outputs(self, data):
        d_idx = getDIdx(data)
        outputs_correct = normalize(data[:, d_idx][None, :].T)
        outputs = np.zeros(outputs_correct.shape)
        return outputs, outputs_correct


    def setup_weights(self, hidden_num, output_num):
        cols_num = self.inputs.shape[1]
        weights1 = np.random.uniform(
            low=(-1 / np.sqrt(self.inputs.shape[1])),
            high=(1 / np.sqrt(self.inputs.shape[1])), size=(cols_num, hidden_num))
        weights2 = np.zeros((hidden_num, output_num))
        return weights1, weights2

    # calculates values of hidden and output layers
    def feed_forward(self):
        #self.inputs = ... # TODO w książce było o obliczeniach na samych inputach
        self.hidden = sigmoid(np.dot(self.inputs, self.weights1))
        self.outputs = sigmoid(np.dot(self.hidden, self.weights2))
        #self.outputs = np.dot(self.hidden, self.weights2) # TODO użyć jakiejkolwiek funkcji aktywacji?

    def gradient_descent(self):
        outputs_errors = loss_der(self.outputs, self.outputs_correct)
        outputs_delta = outputs_errors * sigmoid_der(self.outputs)

        hidden_errors = np.dot(outputs_delta, self.weights2.T)
        hidden_delta = hidden_errors * sigmoid_der(self.hidden)

        return outputs_delta, hidden_delta

    # does whole back propagation
    def back_propagation(self, alpha):
        weights2_delta, weights1_delta = self.gradient_descent()
        self.weights2 -= alpha * np.dot(self.hidden.T, weights2_delta)
        self.weights1 -= alpha * np.dot(self.inputs.T, weights1_delta)

    # trains the network over 'iter' iterations
    def train(self, iter, alpha):
        for i in range(iter):
            self.feed_forward()
            self.back_propagation(alpha)
            self.calculate_error()

    # make prediction based on traning
    def predict(self, data):
        self.inputs = self.setup_inputs(data)
        self.feed_forward()
        return self.outputs

    # calculates error and adds it to errors list
    def calculate_error(self):  # TODO można przemyśleć użycie mse tutaj
        a = self.outputs - self.outputs_correct
        self.errors.append(np.dot(a.T, a)[0][0])
