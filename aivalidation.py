import copy
import math
from random import randrange
from datamanager import *


def get_original_grades(set_to_predict):
    return [row[-1] for row in set_to_predict]


def make_subset(dataset, req_size):
    subset = []
    while len(subset) < req_size:
        index = randrange(len(dataset))
        subset.append(dataset[-1])
        dataset = dataset[:-1]
    return convertToNumpyArray(subset) # TODO zamienić to potem na coś bardziej wydajnego (w funkcjach niżej też)


def two_subset_divide(data, degree):
    learn_data = convertToNumpyArray(data[:int(len(data) * degree)])
    test_data = convertToNumpyArray(data[int(len(data) * degree):])
    return learn_data, test_data


# divides data into given amount (k) of sets
def cross_validation(dataset, k):
    subsets = []

    set_size = int(len(dataset) / k)
    new_set = copy.deepcopy(dataset)

    for i in range(k):
        subset = make_subset(new_set, set_size)
        subsets.append(subset)
    return convertToNumpyArray(subsets)


def count_ratio_guessed_right(original, predicted):
    right = 0
    for i in range(len(original)):
        if original[i] == predicted[i]:
            right += 1
    return right / float(len(original))


def count_rounded_ratio_guessed_right(original_list, predicted_list):
    right = 0
    for original, predicted in zip(original_list, predicted_list):
        if round(float(original)) == round(float(predicted)):
            right += 1
    return right / float(len(original_list))


def count_rights(rights):
    return sum(rights) / len(rights)


def mean_loss(original, predicted, k):
    loses = []
    for i in range(len(original)):
        loss = (original[i] - predicted[i]) ** 2
        loses.append(loss)
    return sum(loses) / len(loses)


def mse_loss(mean_loses, k):
    return 1/k * sum(mean_loses)


def algorithm_validation(test_set, ai):
    # train_set, test_set = divide_type(dataset, degree)

    predicted = ai.predict(test_set)
    predicted = np.interp(predicted, (predicted.min(), predicted.max()), (3, 9))

    actual = get_original_grades(test_set)
    mean_loses = mean_loss(actual, predicted, 2)
    guessed_right = count_ratio_guessed_right(actual, predicted)
    guessed_rounded = count_rounded_ratio_guessed_right(actual, predicted)

    loss_score = mse_loss(mean_loses, 2)

    return loss_score, guessed_right, guessed_rounded, predicted, actual
