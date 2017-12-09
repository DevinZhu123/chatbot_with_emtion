import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
import DataHub as DH
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
from chatbot import *


use_cuda = 0
MAX_LENGTH = 128
EOS_token = 1
SOS_token = 0

def pred(input_variable, encoder, decoder, emoTag, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    input_length = input_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_dim))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    prediction = []

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    prediction.append(SOS_token)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

        # Without teacher forcing: use its own predictions as the next input
    for di in range(1, max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs, emoTag)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        prediction.append(int(ni))

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        if ni == EOS_token:
            break

    return prediction

def predIters(wm, encoder, decoder):
    test_pairs = wm.getBatch()
    for test_pair in test_pairs:
        # training_pair: ((), emocls)
        emoTag = Variable(torch.LongTensor(test_pair[1]))
        input_variable, target_variable = variablesFromPair(test_pair[0])
        prediction = pred(input_variable, encoder, decoder, emoTag)
        print "ori : ", wm.getWordFromIdx(test_pair[0][1][0])
        print "pred: ", wm.getWordFromIdx(prediction)



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
    emoIdx = {mood_dict[i]:i for i in mood_dict}
    emoCls = "../../data/Subtitles/subtitileData/smaller_emotion.txt"
    subtitle = "../../data/Subtitles/subtitileData/smaller.txt"
    test = "../../data/Subtitles/subtitileData/test.txt"
    testEmoCls = "../../data/Subtitles/subtitileData/test_emotion.txt"
    dm = DH.DataManager()
    wm = dm.buildModel(subtitle).buildLookupTabel().data4NN(subtitle, 1)
    wm.setEmotionCls(emoCls)
    wm.setEmoIdx(emoIdx)
    testWm = dm.data4NN(test, 1)
    testWm.setEmotionCls(testEmoCls)
    testWm.setEmoIdx(emoIdx)

    encoderInput_dim, encoderHidden_dim = 10000, 500
    decoderHidden_dim, decoderOutput_dim = 500, 10000
    embedding = nn.Embedding(encoderInput_dim, encoderHidden_dim)
    encoder = EncoderRNN(encoderInput_dim, encoderHidden_dim, embedding)
    decoder = AttnDecoderRNN(decoderHidden_dim, decoderOutput_dim, embedding)
    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder
    encoder.load_state_dict(torch.load('./result/saved_model_lr001_dim500')['en'])
    decoder.load_state_dict(torch.load('./result/saved_model_lr001_dim500')['de'])
    predIters(testWm, encoder, decoder)