import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import DataHub as DH
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


### global setting
teacher_forcing_ratio = 0.5
use_cuda = 0
MAX_LENGTH = 30



class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, _input, hidden):
        embedded = self.embedding(_input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_dim)
        self.emotion_embedding = nn.Embedding(7, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, _input, hidden, encoder_output, encoder_outputs, emotion):
        embedded = self.embedding(_input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = torch.cat((output, self.emotion_embedding(emotion).view(1, -1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def variableFromSentence(sentence):
    # sentence is a list of list of index
    result = Variable(torch.LongTensor(sentence).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(pair[0])
    target_variable = variableFromSentence(pair[1])
    return (input_variable, target_variable)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, emoTag, max_length=MAX_LENGTH):
    # criterion: ?
    # max_length: ?

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs, emoTag)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs, emoTag)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length



def trainIters(encoder, decoder, training_pairs, print_every=1000, plot_every=100, learning_rate=0.01):
    #start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for training_pair in training_pairs:
        # training_pairs: ((), emocls)
        emoTag = training_pair[1]
        input_variable, target_variable = variablesFromPair(training_pair[0])
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, emoTag)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def main(wm, encoder, decoder, epoch):
    for i in range(epoch):
        trainIters(encoder, decoder, wm.getBatch())


if __name__ == "__main__":
    mood_dict = {
        0: 'joy',
        1: 'love',
        2: 'sadness',
        3: 'anger',
        4: 'fear',
        5: 'thankfulness',
        6: 'surprise'
    }
    epoch = 1
    emoIdx = {mood_dict[i]:i for i in mood_dict}
    emoCls = "emotion_class.txt"
    subtitle = "tiny.txt"
    dm = DH.DataManager()
    wm = demo.buildModel(test).buildLookupTabel().data4NN(test, 1)
    wm.setEmotionCls(EMOCLS)
    wm.setEmoIdx(emoIdx)
    encoderInput_dim, encoderHidden_dim = None, None
    decoderHidden_dim, decoderOutput_dim = None, None
    encoder = EncoderRNN(encoderInput_dim, encoderHidden_dim)
    decoder = AttnDecoderRNN(decoderHidden_dim, decoderOutput_dim)
    main(wm, encoder, decoder, epoch)