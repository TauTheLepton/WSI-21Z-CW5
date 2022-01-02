from neuralnetwork import *
from aivalidation import *
from matplotlib import pyplot as plt

# Data
red = 'winequality-red.csv'
white = 'winequality-white.csv'
# Options and settings
hidden_layer_size = 5 # TODO 2 dla normalizacji [0, 1] i sigmoidalnej w ostatniej warstwie, 16 dla [-1,1], bez sigmo i z inną funkcją straty
output_layer_size = 1
era = 1000  # TODO nie wiem jak w tym przypadku tłumaczy się epoka na ang, dlatego dałem era
learning_param = 0.7
# End of options and settings

np.random.seed(1)  # It makes AI predictable


def main():
    test(white, 0.6, learning_param) #0,7 0,3


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
    #for i in range(len(predicted)):
        #print(f'Predicted: {predicted[i]} Original: {actual[i]}')
    print(f'Story of changes: {perceptron.errors}')
    print(f'We missed about (loss): {loss} class, and we guessed right {score_right * 100}%')
    print(f'If we rounded output, we guessed right {score_rounded * 100}%')
    # print(f'Size of predicted set is: {len(predicted)}')
    # print(f'Size of data is {len(data)}')
    plt.plot(perceptron.errors)
    plt.title('Neural Network errors')
    plt.show()


if __name__ == '__main__':
    main()
