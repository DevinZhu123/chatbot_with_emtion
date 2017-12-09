import numpy as np
import itertools as it
import codecs
import re

def loadData(path, seed):
    # loadData function is copied from my homework1
    with open(path, 'r') as file:
        lines = (line[:-1].split(",") for line in file)
        tmp_data = (i for i in lines)
        gen1, gen2 = it.tee(tmp_data)
        dataX = [np.array(i[:-1], dtype=np.float).reshape(28, 28) for i in gen1]
        dataY = [int(i[-1]) for i in gen2]
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

def loadTweets(pathTweets, pathId, seed):
    dataX = []
    dataY = []
    label = [u'love', u'joy', u'thankfulness', u'sadness', u'anger', u'surprise', u'fear']
    with codecs.open(pathTweets, 'r', 'utf-8') as tweets, codecs.open(pathId, 'r', 'utf-8') as ids:
        for tweet, _id in zip(tweets, ids):
            dataX.append(re.findall("\\w+", tweet))
            dataY.append(label.index(re.search("[a-zA-Z]+", _id).group()))
    return dataX, dataY, label

def loadTest(pathTweets, seed):
    dataX = []
    #dataY = []
    #label = [u'love', u'joy', u'thankfulness', u'sadness', u'anger', u'surprise', u'fear']
    with codecs.open(pathTweets, 'r', 'utf-8') as tweets:
        for tweet in tweets:
            dataX.append(re.findall("\\w+", tweet))
    return dataX

class MatchVector:
    def __init__(self, vector_file_path):
        with open(vector_file_path, 'r') as f:
            self.vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                self.vectors[vals[0]] = map(float, vals[1:])
            self.vector_dim = len(self.vectors['the'])



    def get_vector(self, word_list, num_words):
        # num_words = len(word_list)
        ret_matrix = np.zeros((num_words, self.vector_dim))
        for i in xrange(0, num_words):
            tmp = word_list[i].lower()
            if tmp in self.vectors:
                ret_matrix[i] = self.vectors[tmp]
        return ret_matrix

if __name__ == "__main__":
    pathTweets = "/home/lui/CMU/Semester3/10707/RL_chatbot/data/TweetsData/train_1/cleanTrain_1Tweets.txt"
    pathId = "/home/lui/CMU/Semester3/10707/RL_chatbot/data/TweetsData/train_1/tweets_with_tags_train_1_IdMoods.txt"
    X, Y, label = loadTweets(pathTweets, pathId, seed=1)
    vec = "/home/lui/CMU/Semester3/10707/proj/lookupTable/glove.6B/glove.6B.50d.txt"
    demo = MatchVector(vec)
    print X[0]
    vecX = demo.get_vector(X[0], len(X[0]))
    for i in range(len(X[0])):
        print X[0][i], np.sum(vecX[i])
