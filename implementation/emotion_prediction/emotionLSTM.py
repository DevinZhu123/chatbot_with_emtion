import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
import pickle


class EmotionLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_dim, n_layer=1):
        super(EmotionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.target_dim = target_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim, n_layer)
        self.hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(hidden_dim, target_dim)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, _input, seq_len):
        _input = _input.view(seq_len, 1, -1)
        _, (hn, cn) = self.LSTM(_input, self.hidden)
        hn = hn.view(-1, self.hidden_dim)
        output = self.hidden2tag(hn)
        output = F.log_softmax(output)
        return output


def loadData(path_to_tweet, path_to_tag, path_to_word_vec):
    # load
    dataX, dataY = utils.load_tweet_data(path_to_tweet, path_to_tag, path_to_word_vec)
    num_samples = len(dataX)
    tX = dataX[: int(num_samples*0.9)]
    tY = dataY[0: int(num_samples*0.9)]
    vX = dataX[int(num_samples*0.9):]
    vY = dataY[int(num_samples*0.9):]
    return tX, tY, vX, vY


def testDemoTweets():
    path_to_tweet = '../../data/TweetsData/train_1/cleanTrain_1TweetsWithTags.txt'
    path_to_tag = '../../data/TweetsData/train_1/tweets_with_tags_train_1_IdMoods.txt'
    path_to_word_vec = '../../data/TweetsData/glove.6B.50d.txt'
    dataX, dataY, validX, validY = loadData(path_to_tweet, path_to_tag, path_to_word_vec)

    model = EmotionLSTM(input_dim=50, hidden_dim=200, target_dim=7)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001)

    torch.save(model.state_dict(), './trained_model')

    # train model
    losses = []
    iters = 0
    num_samples = len(dataX)
    accuracy_list = []
    for epoch in range(1):
        print "epoch: ", epoch+1

        for i in range(num_samples):
            iters += 1
            _input = dataX[i]
            seq_len = _input.shape[0]
            _input = Variable(torch.from_numpy(_input)).float()
            tag = Variable(torch.from_numpy(np.array([dataY[i]])))
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_score = model(_input, seq_len)
            loss = loss_function(tag_score, tag)
            loss.backward()
            optimizer.step()

            if iters % 1000 == 0:
                num_tests = len(validX)
                count = 0
                for j in range(num_tests):
                    _input = validX[j]
                    seq_len = _input.shape[0]
                    model.hidden = model.init_hidden()
                    _input = Variable(torch.from_numpy(_input)).float()
                    tag_score = model(_input, seq_len)
                    pred_tag = np.argmax(tag_score.data.numpy())
                    if pred_tag == validY[j]:
                        count += 1
                print 'test accuracy: '+str(count*1.0/num_tests)
                accuracy_list.append(1 - count*1.0/num_tests)
    torch.save(model.state_dict(), './trained_model1')
    with open('./accuracy.csv', 'wb') as file:
        pickle.dump(accuracy_list, file)


def plot():
    with open('./accuracy.csv', 'rb') as file:
        acc = pickle.load(file)
    num_dp = len(acc)
    x = [i*1000 for i in range(1, num_dp+1)]
    plt.plot(x, acc)
    plt.xlabel('number of samples')
    plt.ylabel('classification error')
    plt.show()


if __name__ == "__main__":
    plot()
