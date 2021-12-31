import csv
import math
import numpy as np
from numpy.core.fromnumeric import size


# returns list of tuples as data with onli the data and header is one tuple with names of elements in tuples
def readFile(file_name):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file, delimiter=';')
        data = [list(row) for row in reader]
        header = data[0]
        data.remove(data[0])
    return data, header


# was supposed to convert every element of data from string to float, but it doesn't matter
def convertDataToFloat(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    return data


#temporary function to convert data to numpy array
def convertToNumpyArray(data):
    return np.asarray(data)


# divides data into learn data and test data with given coefficient
def divideLearnTestData(data, coef):
    learn_data = data[:int(len(data)*coef)]
    test_data = data[int(len(data)*coef):]
    return learn_data, test_data


# divides data into given amount (k) of sets
def divideDataIntoSets(data, k):
    sets = []
    set_length = math.trunc(len(data) / k)
    for i in range(k):
        sets.append(data[i*set_length:(i+1)*set_length])
    return sets


# merges all sets form list 'sets' without set 'exclude_set' into one set
def mergeSets(sets, exclude_set):
    merged = []
    for set in sets:
        if set != exclude_set:
            for item in set:
                merged.append(item)
    return merged


# returns index of element d, so in this case the last one
def getDIdx(data):
    return len(data[0])-1


# creates a list containing every different d once
def getDList(data):
    D = []
    d_idx = getDIdx(data)
    for item in data:
        is_d = False
        for d in D:
            if item[d_idx] == d:
                is_d = True
        if not is_d:
            D.append(item[d_idx])
    return D


# Array normalization
def normalize(array: np.array):
    return np.interp(array, (array.min(), array.max()), (0, +1))
    #return np.interp(array, (array.min(), array.max()), (-1, +1))


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
    def feedForward(self): # według książki przez sigmoid trzeba przepuścić tylko raz
        #self.inputs = ...
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights1))
        self.outputs = self.sigmoid(np.dot(self.hidden, self.weights2))

    # calculates error and adds it to errors list
    def calculateError(self):
        a = self.outputs - self.outputs_correct

        value = np.interp(self.outputs, (self.outputs.min(), self.outputs.max()), (3, 9))
        self.values.append(value)
        self.errors.append(np.dot(a.T, a)[0][0])

    # does whole back propagation (idk how it works)
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


# tests neural network
def test(filename, coef):
    # imports data from file
    data, header = readFile(filename)
    data = convertDataToFloat(data)
    data = convertToNumpyArray(data)
    learn_data, test_data = divideLearnTestData(data, coef)
    # tests some basics
    learn_data = np.array(
        [[2, 3, 4],
        [534, 12, 34],
        [7, 7, 12],
        [45, 678, 12],
        [345, 678, 12]]
    )

    NN = NeuralNetwork(data, 2, 1)
    NN.train(4000)
    print(NN.errors)
    #print(NN.values)


def main():
    red = 'winequality-red.csv'
    white = 'winequality-white.csv'
    test(red, 0.6)


if __name__ == '__main__':
    main()
