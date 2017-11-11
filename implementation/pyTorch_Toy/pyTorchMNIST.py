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


def testDemoMNIST():
    model = demoLSTM(batch_size=1, embedding_dim=28, hidden_dim=100, targetSize=10)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    addr = "path/of/train"
    dataX, dataY = loadData(addr)
    # test loaded data
    print "loaded data", dataX[0:1].size()
    # test model
    tag_score = model(dataX[0:1])
    loss = loss_function(tag_score, dataY[0:1])
    print "model score, softmax", tag_score, "loss", loss.data
    # train model
    losses = []
    iters = 0
    for epoch in range(10):
        print "epoch: ", epoch+1
        for i in range(dataX.size()[0]):
            iters += 1
            img = dataX[i:i+1]
            tag = dataY[i:i+1]
            model.zero_grad()
            model.hidden = model.init_LSTMhidden()
            tag_score = model(img)
            loss = loss_function(tag_score, tag)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.numpy()[0])
            if iters % 1000 == 0:
                print "loss", loss.data.numpy()[0]
    print "finish training"
    print "test overfit"
    tag_score = model(dataX[0:1])
    d1 = 0
    d2 = 0
    for i in range(dataX.size()[0]):
        d2 += 1
        img = dataX[i:i+1]
        tag = dataY[i].data.numpy()[0]
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "train accuracy", d1 * 1.0 / d2
    print "test validation"
    addr = "path/of/validation"
    validX, validY = loadData(addr)
    d1 = 0
    d2 = 0
    for i in range(validX.size()[0]):
        d2 += 1
        img = validX[i:i+1]
        tag = validY[i].data.numpy()[0]
        model.zero_grad
        model.hidden = model.init_LSTMhidden()
        tag_score = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
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
