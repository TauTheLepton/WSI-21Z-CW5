 import math
from datamanager import *


# class doing all neural network calculations and storing its data
class NeuralNetwork:
    def __init__(self, learn_data, hidden_num, output_num):
        """
        learn_data: array-like with data to learn from, size of learn data decide about number of input neurons
        hidden_num: number of neurons in hidden layer of neural network
        output_num: number of neurons in output layer of neural network
        """

        d_idx = getDIdx(learn_data)
        learn_data = np.array(learn_data)
        self.inputs = normalize(learn_data[:, :d_idx])
        self.outputs_correct = normalize(learn_data[:, d_idx][None, :].T)
        self.outputs = np.zeros(self.outputs_correct.shape)
        cols_num = self.inputs.shape[1]
        self.weights1 = np.random.uniform(low=(-1/np.sqrt(self.inputs.shape[1])), high=(1/np.sqrt(self.inputs.shape[1])), size=(cols_num, hidden_num))
        self.weights2 = np.zeros((hidden_num, output_num))
        self.errors = []
        self.values = []

    # calculates value of sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # calculates value of derivative of sigmoid function
    def sigmoidDer(self, x):
        return x * (1 - x)

    # calculates values of hidden and output layers
    def feedForward(self): # TODO check if sigmoid function should be also in output layer (book said it shouldnt)
        #self.inputs = ...
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.outputs = self.sigmoid(np.dot(self.hidden, self.weights2))

    # calculates error and adds it to errors list
    def calculateError(self):
        a = self.outputs - self.outputs_correct

        value = np.interp(self.outputs, (self.outputs.min(), self.outputs.max()), (3, 9))
        self.values.append(value) # to throw out
        self.errors.append(np.dot(a.T, a)[0][0])

    # does whole back propagation
    def backPropagation(self):
        d_weights2 = np.dot(self.hidden.T, (2 * (self.outputs_correct - self.outputs) * self.sigmoidDer(self.outputs)))
        d_weights1 = np.dot(self.inputs.T, (np.dot(2 * (self.outputs_correct - self.outputs) \
            * self.sigmoidDer(self.outputs), self.weights2.T) * self.sigmoidDer(self.hidden)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # trains the network over 'iter' iterations
    def train(self, iter):
        for i in range(iter):
            self.feedForward()
            self.backPropagation()
            self.calculateError()

    # make prediction based on traning
    def predict(self, data):
        self.inputs = normalize(data[:, :11]) # TODO zmienić by 11 nie była hardkodowana tutaj
        self.feedForward()
        return self.outputs







