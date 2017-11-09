import numpy as np
import time


class MatchVector:
    def __init__(self, vector_file_path):
        with open(vector_file_path, 'r') as f:
            self.vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                self.vectors[vals[0]] = map(float, vals[1:])
            self.vector_dim = len(self.vectors['the'])



    def get_vector(self, word_list):
        num_words = len(word_list)
        ret_matrix = np.zeros((num_words, self.vector_dim))
        for i in xrange(0, num_words):
            if word_list[i] in self.vectors:
                ret_matrix[i] = self.vectors[word_list[i]]
        return ret_matrix



start_time = time.time()
mv = MatchVector('../data/vector/glove.6B/glove.6B.50d.txt')
load_time = time.time()
print ("--- loading vector file: %s seconds ---" % (load_time - start_time))
mv.get_vector(['the', 'hell', 'is', 'this'])
query_time1 = time.time()
print ("--- get one sentence: %s seconds ---" % (query_time1 - load_time))
