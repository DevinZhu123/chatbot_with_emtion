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
        return (Variable(torch.zeros(self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.batch_size, self.hidden_dim)))

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


def loadData(addr):
    # load MNIST
    dataX, dataY = utils.loadData(addr, 1)
    tX = torch.from_numpy(dataX).float()
    tY = torch.from_numpy(dataY)
    return Variable(tX), Variable(tY)

def loadTweet(pathTweets, pathId):
    dataX, dataY, label = utils.loadTweets(pathTweets, pathId, seed=1)
    vec = "/home/lui/CMU/Semester3/10707/proj/lookupTable/glove.6B/glove.6B.50d.txt"
    wm = utils.MatchVector(vec)
    X = []
    Y = Variable(torch.LongTensor(dataY))
    for i in range(len(dataX)):
        tmp = wm.get_vector(dataX[i], len(dataX[i]))
        print tmp.shape, dataX[i], i
        X.append(Variable(torch.from_numpy(tmp).float()))
    return X, Y, label, dataX

def loadTest(pathTweets):
    dataX = utils.loadTest(pathTweets, seed=1)
    vec = "/home/lui/CMU/Semester3/10707/proj/lookupTable/glove.6B/glove.6B.50d.txt"
    wm = utils.MatchVector(vec)
    X = []
    for i in range(len(dataX)):
        tmp = wm.get_vector(dataX[i], len(dataX[i]))
        print tmp.shape, dataX[i], i
        X.append(Variable(torch.from_numpy(tmp).float()))
    return X, dataX


def testDemoTWEET():
    model = demoLSTM(batch_size=1, embedding_dim=50, hidden_dim=100, targetSize=7)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    pathTweets = "/your/tweet/data.txt"
    pathId = "/your/tweets_with_tags_train_1_IdMoods.txt"
    testData = "/your/clearEmotion.txt"
    # dataX, a list; dataY, pytorch Variable; label, a list; sentences, tweets;
    dataX, dataY, label, sentences = loadTweet(pathTweets, pathId)
    testX, testSentence = loadTest(testData)
    # test loaded data
    print "loaded data", dataX[0].size()
    print "sample tweets:", " ".join(sentences[0])
    print "test model ......"
    tag_score = model(dataX[0].view(1, dataX[0].size()[0], dataX[0].size()[1]))
    loss = loss_function(tag_score, dataY[0:1])
    print "model score, softmax", tag_score.data.numpy(), "loss", loss.data.data.numpy()
    print "testData:", testSentence[0]
    print "Done! "
    print "train model"
    losses = []
    iters = 0
    length = 8 * len(dataX) / 10
    for epoch in range(1):
        print "epoch: ", epoch+1
        for i in range(length):
            iters += 1
            img = dataX[i].view(1, dataX[i].size()[0], dataX[i].size()[1])
            tag = dataY[i:i+1]
            model.zero_grad()
            model.hidden = model.init_LSTMhidden()
            tag_score = model(img)
            loss = loss_function(tag_score, tag)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.numpy()[0])
            if iters % 1000  == 0:
                print "loss", loss.data.numpy()[0]
    print "finish training"
    print "test overfit"
    #tag_score = model(dataX[0:1])
    d1 = 0
    d2 = 0
    for i in range(length):
        d2 += 1
        img = dataX[i].view(1, dataX[i].size()[0], dataX[i].size()[1])
        tag = dataY[i].data.numpy()[0]
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "train accuracy", d1 * 1.0 / d2

    print "test validation"
    d1 = 0
    d2 = 0
    for i in range(length, len(dataX)):
        d2 += 1
        img = dataX[i].view(1, dataX[i].size()[0], dataX[i].size()[1])
        tag = dataY[i].data.numpy()[0]
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "validation accuracy", d1 * 1.0 / d2

    print "test prediction"
    d1 = 0
    d2 = 0
    for i in range(len(testX)):
        d2 += 1
        img = testX[i].view(1, testX[i].size()[0], testX[i].size()[1])
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score, alignCoef = model(img)
        pred = np.argmax(tag_score.data.numpy())
        print "test sentence:   ", testSentence[i]
        print "test prediction: ", label[pred]
        print "test attention:  ", alignCoef.data.numpy()
    print "validation accuracy", d1 * 1.0 / d2
    return losses

def plot(data):
    x = range(1, len(data)+1)
    plt.plot(x, data, '--.g', label="trainloss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    losses = testDemoTWEET()
    plot(losses)
