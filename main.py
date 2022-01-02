from neuralnetwork import *
from aivalidation import *

# Data
red = 'winequality-red.csv'
white = 'winequality-white.csv'
# Options and settings
hidden_layer_size = 2 # TODO 2 dla normalizacji [0, 1] i sigmoidalnej w ostatniej warstwie, 16 dla [-1,1], bez sigmo i z inną funkcją straty
output_layer_size = 1
era = 12000  # TODO nie wiem jak w tym przypadku tłumaczy się epoka na ang, dlatego dałem era
learning_param = 0.7
# End of options and settings

np.random.seed(1)  # TODO seed dodany do testów, usunąć go potem


def main():
    test(red, 0.6, learning_param) #0,7 0,3


# tests neural network
def test(filename, coef, alpha):
    # imports data from file
    data, header = read_file(filename)
    data = convertToNumpyArray(convertDataToFloat(data))

    learn_data, test_data = two_subset_divide(data, coef)

    perceptron = NeuralNetwork(learn_data, hidden_layer_size, output_layer_size)
    perceptron.train(era, alpha)
    loss, score_right, score_rounded, predicted, actual = algorithm_validation(test_data, perceptron)

    # print('Compare between predicted and original:')
    # for i in range(len(predicted)):
    #     print(f'Predicted: {predicted[i]} Original: {actual[i]}')
    print(f'Our loss is: {loss}, and our score is {score_right * 100}')
    print(f'Score rounded is {score_rounded * 100}')
    # print(f'Size of predicted set is: {len(predicted)}')
    # print(f'Size of data is {len(data)}')


if __name__ == '__main__':
    main()
