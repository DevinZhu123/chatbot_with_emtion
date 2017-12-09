import numpy as np
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import matplotlib.pyplot as plt


class EmotionClassifier(nn.Module):
    def __init__(self,
                 batch_size,
                 embedding_dim, 
                 encode_hiddenUnit, 
                 decode_hiddenUnit, 
                 targetSize):
        super(EmotionClassifier, self).__init__()
        self.batch_size = batch_size
        self.encode_hiddenUnit = encode_hiddenUnit
        #self.decode_hiddenUnit = decode_hiddenUnit
        self.encodeLSTM = nn.LSTMCell(embedding_dim, encode_hiddenUnit)
        #self.decodeRNN = nn.RNNCell(embedding_dim, decode_hiddenUnit)
        self.encodeHidden = self.init_encodeLSTMhidden()
        #self.decodeHidden = self.init_decodeLSTMhidden()
        self.attentionA = nn.Linear(encode_hiddenUnit, 1)
        self.hidden2tag = nn.Linear(embedding_dim, targetSize)
        

    def init_encodeLSTMhidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.batch_size, self.encode_hiddenUnit)),
                Variable(torch.zeros(self.batch_size, self.encode_hiddenUnit)))

    def init_decodeLSTMhidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return Variable(torch.zeros(self.batch_size, self.decode_hiddenUnit))

    def init_RNNhidden(self):
        return Variable(torch.zeros(self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        # sentence: variable matrix contains batch sized sentences, each sentence
        #           is a list of word vectors.
        #           shape: [batch_size, words, dim]
        encoderHiddens = []
        batch, wordNum, dim = sentence.size()
        ec, eh = self.encodeHidden
        words = [sentence[:,i,:] for i in xrange(wordNum)]
        for i in xrange(wordNum):
            ec, eh = self.encodeLSTM(words[i], (ec, eh))
            encoderHiddens.append(eh) # eh, shape [batch, hidden]
        # compute the attention weights
        alignCoef = self._alignCoef(encoderHiddens)
        weightedWords = self._attention(alignCoef, words)
        #dh = torch.cat([eh, weightedWords], dim=1)
            
        # output
        cls_space = self.hidden2tag(weightedWords)
        cls_score = F.log_softmax(cls_space)
        return cls_score, alignCoef

    def _alignCoef(self, x):
        # h is the hidden state at t-1 steps
        # x is a list of tensors, hidden states of encoder LSTM
        rst = []
        for xi in x:
            a = self.attentionA(xi)
            rst.append(a)
        # stack to 
        e = torch.stack(rst, dim=1).view(self.batch_size, len(rst))
        alignCoef = torch.nn.Softmax()(e)
        return alignCoef


    def _attention(self, alignCoef, x):
        iS = range(len(x))
        return sum(map(lambda i: alignCoef[:, i] * x[i], iS))


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


def testAttentionMNIST():
    model = EmotionClassifier(batch_size=1,
                              embedding_dim=28, 
                              encode_hiddenUnit=100,    
                              decode_hiddenUnit=100, 
                              targetSize=10)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    addr = "/home/lui/CMU/Semester3/10707/hw2/data/digitstrain.txt"
    dataX, dataY = loadData(addr)
    # test loaded data
    print "loaded data", dataX[0].size()
    tag_score, alignCoef = model(dataX[0:1])
    loss = loss_function(tag_score, dataY[0:1])
    print "model score, softmax", tag_score, "loss", loss.data
    print "attention coef", alignCoef
    # train model
    losses = []
    iters = 0
    for epoch in range(5):
        print "epoch: ", epoch+1
        for i in range(dataX.size()[0]):
            iters += 1
            img = dataX[i:i+1]
            tag = dataY[i:i+1]
            model.zero_grad()
            model.encodeHidden = model.init_encodeLSTMhidden()
            model.decodeHidden = model.init_decodeLSTMhidden()
            tag_score, _ = model(img)
            #print tag_score
            loss = loss_function(tag_score, tag)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.numpy()[0])
            if iters % 500 == 0:
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
        model.encodeHidden = model.init_encodeLSTMhidden()
        model.decodeHidden = model.init_decodeLSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "train accuracy", d1 * 1.0 / d2
    print "test validation"
    addr = "/home/lui/CMU/Semester3/10707/hw2/data/digitsvalid.txt"
    validX, validY = loadData(addr)
    d1 = 0
    d2 = 0
    for i in range(validX.size()[0]):
        d2 += 1
        img = validX[i:i+1]
        tag = validY[i].data.numpy()[0]
        model.zero_grad
        model.encodeHidden = model.init_encodeLSTMhidden()
        model.decodeHidden = model.init_decodeLSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "validation accuracy", d1 * 1.0 / d2
    return losses


def testDemoMNIST():
    model = demoLSTM(batch_size=1, embedding_dim=28, hidden_dim=100, targetSize=10)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    addr = "/home/lui/CMU/Semester3/10707/hw2/data/digitstrain.txt"
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
    addr = "/home/lui/CMU/Semester3/10707/hw2/data/digitsvalid.txt"
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


def testDemoTWEET():
    #model = demoLSTM(batch_size=1, embedding_dim=50, hidden_dim=100, targetSize=7)
    model = EmotionClassifier(batch_size=1,
                              embedding_dim=50, 
                              encode_hiddenUnit=100,    
                              decode_hiddenUnit=100, 
                              targetSize=7)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    pathTweets = "/home/lui/CMU/Semester3/10707/proj/TweetsData/tweetWithTag.txt"
    pathId = "/home/lui/CMU/Semester3/10707/RL_chatbot/data/TweetsData/train_1/tweets_with_tags_train_1_IdMoods.txt"
    testData = "/home/lui/CMU/Semester3/10707/proj/clearEmotion.txt"
    dataX, dataY, label, sentences = loadTweet(pathTweets, pathId)
    testX, testSentence = loadTest(testData)
    # test loaded data
    print "loaded data", dataX[0].size()
    print "sample tweets:", " ".join(sentences[0])
    # test model
    #tag_score = model(dataX[0].view(1, dataX[0].size()[0], dataX[0].size()[1]))
    tag_score, alignCoef = model(dataX[0].view(1, dataX[0].size()[0], dataX[0].size()[1]))
    loss = loss_function(tag_score, dataY[0:1])
    print "model score, softmax", tag_score.data.numpy(), "loss", loss.data.data.numpy()
    print "attention: ", alignCoef.data.numpy()
    print "testData:", testSentence[0]
    # train model
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
            model.encodeHidden = model.init_encodeLSTMhidden()
            ##model.decodeHidden = model.init_decodeLSTMhidden()
            #model.hidden = model.init_LSTMhidden()
            tag_score, _ = model(img)
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
        model.encodeHidden = model.init_encodeLSTMhidden()
        ##model.decodeHidden = model.init_decodeLSTMhidden()
        #model.hidden = model.init_LSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())
        if pred == tag:
            d1 += 1
    print "train accuracy", d1 * 1.0 / d2

    print "test validation"
    #tag_score = model(dataX[0:1])
    d1 = 0
    d2 = 0
    for i in range(length, len(dataX)):
        d2 += 1
        img = dataX[i].view(1, dataX[i].size()[0], dataX[i].size()[1])
        tag = dataY[i].data.numpy()[0]
        model.zero_grad
        model.encodeHidden = model.init_encodeLSTMhidden()
        ##model.decodeHidden = model.init_decodeLSTMhidden()
        #model.hidden = model.init_LSTMhidden()
        tag_score, _ = model(img)
        pred = np.argmax(tag_score.data.numpy())

        if pred == tag:
            d1 += 1
    print "validation accuracy", d1 * 1.0 / d2

    print "test validation"
    #tag_score = model(dataX[0:1])
    d1 = 0
    d2 = 0
    for i in range(len(testX)):
        d2 += 1
        img = testX[i].view(1, testX[i].size()[0], testX[i].size()[1])
        #tag = dataY[i].data.numpy()[0]
        model.zero_grad
        model.encodeHidden = model.init_encodeLSTMhidden()
        ##model.decodeHidden = model.init_decodeLSTMhidden()
        #model.hidden = model.init_LSTMhidden()
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
    #losses = testAttentionMNIST()
    #losses = testDemoMNIST()
    losses = testDemoTWEET()
    plot(losses)
