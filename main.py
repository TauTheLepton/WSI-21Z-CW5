from neuralnetwork import *
from aivalidation import *


# Options and settings
red = 'winequality-red.csv'
white = 'winequality-white.csv'
hidden_layer_size = 2
output_layer_size = 1
epoki = 12000  # TODO zmienić na na angielski (nie wiem jak to po angielsku będzie)
# End of options and settings


def main():
    test(red, 0.6)


# tests neural network
def test(filename, coef):
    # imports data from file
    data, header = read_file(filename)
    data = convertDataToFloat(data)
    data = convertToNumpyArray(data)

    perceptron = NeuralNetwork(data, hidden_layer_size, output_layer_size)
    perceptron.train(epoki)
    loss, score, predicted, actual = algorithm_validation(data, two_subset_divide, coef, perceptron)

    print('Compare between predicted and original:')
    for i in range(len(predicted)):
        print(f'Predicted: {predicted[i]} Original: {actual[i]}')
    print(f'Our loss is: {loss}, and our score is {score * 100}')
    print(f'Size of predicted set is: {len(predicted)}')
    print(f'Size of data is {len(data)}')


if __name__ == '__main__':
    main()
