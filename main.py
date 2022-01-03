from neuralnetwork import *
from aivalidation import *
from matplotlib import pyplot as plt

from settings import *

from sklearn.datasets import load_iris


np.random.seed(1)  # It makes AI predictable

def main():
    test(use_file, 0.6, learning_param)
    test_iris(0.6, learning_param)

# tests neural network with given data file
def test(filename, coef, alpha):
    # imports data from file
    data, header = read_file(filename)
    data = convertToNumpyArray(convertDataToFloat(data))

    learn_data, test_data = two_subset_divide(data, coef)

    perceptron = NeuralNetwork(learn_data, hidden_layer_size, output_layer_size, bias)
    perceptron.train(era, alpha)
    loss, score_right, score_rounded, predicted, actual = algorithm_validation(test_data, perceptron, (3, 9))

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

# tests neural network with iris data set
def test_iris(coef, alpha):
    data = load_iris(return_X_y=True)
    data = np.concatenate((data[0], data[1][:, None]), axis=1)

    learn_data, test_data = two_subset_divide(data, coef)

    perceptron = NeuralNetwork(learn_data, hidden_layer_size, output_layer_size, bias)
    perceptron.train(era, alpha)
    loss, score_right, score_rounded, predicted, actual = algorithm_validation(test_data, perceptron, (0, 2))

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
