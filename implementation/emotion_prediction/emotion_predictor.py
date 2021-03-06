from emotionLSTM import EmotionLSTM
import torch
from utils import MatchVector
import re
from torch.autograd import Variable
import numpy as np


mood_dict = {
    0: 'joy',
    1: 'love',
    2: 'sadness',
    3: 'anger',
    4: 'fear',
    5: 'thankfulness',
    6: 'surprise'
}


class EmotionPredictor:

    def __init__(self, model_path='./trained_model', path_to_word_vec='../../data/TweetsData/glove.6B.50d.txt'):
        self.model = EmotionLSTM(input_dim=50, hidden_dim=200, target_dim=7)
        self.model.load_state_dict(torch.load(model_path))
        self.mv = MatchVector(path_to_word_vec)

    def predict(self, sentence):
        sentence = " ".join(re.findall('\\w+', sentence.lower()))
        words = sentence.split(' ')
        _input = self.mv.get_matrix(words, len(words))
        seq_len = _input.shape[0]
        self.model.hidden = self.model.init_hidden()
        _input = Variable(torch.from_numpy(_input)).float()
        tag_score = self.model(_input, seq_len)
        pred_tag = np.argmax(tag_score.data.numpy())
        return mood_dict[pred_tag]


if __name__ == "__main__":
    ep = EmotionPredictor()
    print ep.predict("It s so frustrating working with him .")
    print ep.predict('I got everything I ever wanted. I feel so blessed .')
    print ep.predict('I so pissed . Roger just stabbed me in the back .')
    print ep.predict('I was so frustrated , I stopped caring about the outcome .')
    print ep.predict('It feels so good taking a long vacation .')
    emotions = []

    """
    with open('../../data/Subtitles/subtitileData/test.txt', 'r') as file:
        for line in file:
            line = " ".join(line.split(" "))[1: -1]
            emotions.append(ep.predict(line))
    with open('../../data/Subtitles/subtitileData/test_emotion.txt', 'w') as file:
        for emotion in emotions:
            file.write(emotion+'\n')
    """