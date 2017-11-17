import numpy as np
import re

mood_dict = {
    'joy': 0,
    'love': 0,
    'sadness': 1,
    'anger': 1,
    'fear': 1,
    'thankfulness': 2,
    'surprise': 2
}

class MatchVector:
    def __init__(self, vector_file_path):
        with open(vector_file_path, 'r') as f:
            self.vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                self.vectors[vals[0]] = map(float, vals[1:])
            self.vector_dim = len(self.vectors['the'])



    def get_matrix(self, word_list, num_words):
        words_len = len(word_list)
        ret_matrix = np.zeros((num_words, self.vector_dim))
        for i in xrange(0, num_words):
            if i < words_len and word_list[i] in self.vectors:
                ret_matrix[i] = self.vectors[word_list[i]]
        return ret_matrix


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


def load_tweet_data(path_to_tweet, path_to_tag, path_to_word_vec, seed=1):
    mv = MatchVector(path_to_word_vec)

    sentences = []
    max_sentence_size = 0
    with open(path_to_tweet, 'r') as f:
        for line in f:
            sentence = " ".join(re.findall('\\w+', line))
            words = sentence.split(' ')
            max_sentence_size = max(len(words), max_sentence_size)
            sentences.append(words)

    # max_sentence_size = max_sentence_size * 4 / 5
    max_sentence_size = 28
    dataX = [mv.get_matrix(words, max_sentence_size) for words in sentences]
    dataX = np.array(dataX)

    dataY = []
    with open(path_to_tag, 'r') as f:
        for line in f:
            mood = line.split()[1]
            dataY.append(mood_dict[mood])
    dataY = np.array(dataY, dtype=int)

    return dataX, dataY
    






