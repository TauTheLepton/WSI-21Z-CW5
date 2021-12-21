import csv
import math
import numpy as np

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

class NeuralNetwork:

    def __init__(self, learn_data):
        pass

def main():
    red = 'winequality-red.csv'
    white = 'winequality-white.csv'
    data, header = readFile(red)

if __name__ == '__main__':
    main()
