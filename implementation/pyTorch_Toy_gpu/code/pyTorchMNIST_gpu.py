import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import matplotlib.pyplot as plt


class demoLSTM(nn.Module):
    def __init__(self,
                 batch_size,
                 embedding_dim, 
                 hidden_dim,
                 targetSize):
        super(demoLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden = self.init_LSTMhidden()
        self.hidden2tag = nn.Linear(hidden_dim, targetSize)

    def init_LSTMhidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(self.batch_size, self.hidden_dim).cuda()))

    def init_RNNhidden(self, hidden_dim):
        return Variable(torch.zeros(self.batch_size, hidden_dim))

    def forward(self, _input):
        # _input: shape [batch, 28, 28]
        batch, dimR, dimC = _input.size()
        ec, eh = self.hidden
        rows = [_input[:,i,:] for i in xrange(dimR)]
        for i in rows:
            ec, eh = self.LSTM(i, (ec, eh))
        cls_space = self.hidden2tag(eh)
        cls_score = F.log_softmax(cls_space)
        return cls_score


def loadData(path_to_tweet, path_to_tag, path_to_word_vec):
    # load MNIST
    dataX, dataY = utils.load_tweet_data(path_to_tweet, path_to_tag, path_to_word_vec)
    tX = torch.from_numpy(dataX).float()
    tY = torch.from_numpy(dataY)
    return Variable(tX.cuda()), Variable(tY.cuda())


def testDemoTweets():
    batch_size = 10
    model = demoLSTM(batch_size=batch_size, embedding_dim=50, hidden_dim=200, targetSize=4).cuda()
    loss_function = nn.NLLLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # addr = "../data/digitstrain.txt"
    path_to_tweet = '../../../data/TweetsData/train_1/cleanTrain_1Tweets.txt'
    path_to_tag = '../../../data/TweetsData/train_1/tweets_with_tags_train_1_IdMoods.txt'
    path_to_word_vec = '../../../data/vector/glove.6B/glove.6B.50d.txt'
    dataX, dataY = loadData(path_to_tweet, path_to_tag, path_to_word_vec)
    # test loaded data
    print "loaded data", dataX[0:10].size()
    # test model
    tag_score = model(dataX[0:10])
    loss = loss_function(tag_score, dataY[0:10])
    print "model score, softmax", tag_score, "loss", loss.data
    # train model
    losses = []
    iters = 0
    for epoch in range(50):
        print "epoch: ", epoch+1
        for i in range(dataX.size()[0]/batch_size):
            iters += 1
            img = dataX[i*batch_size:(i+1)*batch_size]
            tag = dataY[i*batch_size:(i+1)*batch_size]
            model.zero_grad()
            model.hidden = model.init_LSTMhidden()
            tag_score = model(img)
            loss = loss_function(tag_score, tag)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu().numpy()[0])
            if iters % 1000 == 0:
                print "loss", loss.data.cpu().numpy()[0]
    print "finish training"
    print "test overfit"
    # tag_score = model(dataX[0:1])
    d1 = 0
    d2 = 0
    for i in range(dataX.size()[0]/batch_size):
        d2 += batch_size
        img = dataX[i*batch_size:(i+1)*batch_size]
        tag = dataY[i*batch_size:(i+1)*batch_size].data.cpu().numpy()
        model.zero_grad()
        model.hidden = model.init_LSTMhidden()
        tag_score = model(img)
        # print tag_score.data.cpu().numpy()
        pred = np.argmax(tag_score.data.cpu().numpy(), axis=1)
        # print(pred.shape, tag.shape)
        assert pred.shape == tag.shape, 'prediction shape error.'
        d1 += np.sum(pred == tag)
    print "train accuracy", d1 * 1.0 / d2

    """
    print "test validation"
    addr = "../data/digitsvalid.txt"
    validX, validY = loadData(addr)
    d1 = 0
    d2 = 0
    for i in range(validX.size()[0]):
        d2 += 1
        img = validX[i:i+1]
        tag = validY[i].data.cpu().numpy()[0]
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score = model(img)
        pred = np.argmax(tag_score.data.cpu().numpy())
        if pred == tag:
            d1 += 1
    print "validation accuracy", d1 * 1.0 / d2
    """

    return losses

def plot(data):
    x = range(1, len(data)+1)
    plt.plot(x, data, '--.g', label="trainloss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    losses = testDemoTweets()
    plot(losses)
