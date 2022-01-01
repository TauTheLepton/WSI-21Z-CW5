import csv
import numpy as np


# returns list of tuples as data with onli the data and header is one tuple with names of elements in tuples
def read_file(file_name):
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


# Array normalization
def normalize(array: np.array):
    return np.interp(array, (array.min(), array.max()), (0, 1)) # TODO [0,1] czy [-1, 1] ?


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