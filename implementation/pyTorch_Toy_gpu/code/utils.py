import numpy as np


def loadData(path, seed=1):
    # loadData function is copied from my homework1
    with open(path, 'r') as file:
        lines = (line[:-1].split(",") for line in file)
        tmp_data = [i for i in lines]
        dataX = [np.array(i[:-1], dtype=np.float).reshape(28, 28) for i in tmp_data]
        dataY = [int(i[-1]) for i in tmp_data]
    dataX = np.array(dataX)
    dataY = np.array(dataY) 
    indices = np.arange(dataX.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    dataX = dataX[indices, :]
    dataY = dataY[indices]
    return dataX, dataY

def toOneHot(digit, length):
    # toOneHot function is copied from my homework1
    digit = int(digit)
    tmp = np.zeros(length, dtype=np.int)
    tmp[digit] = 1
    return tmp
