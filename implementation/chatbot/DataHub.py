from collections import defaultdict as ddict
from io import open
import re
import random
import itertools as it


class DataManager:
    def __init__(self, vocabSize=10000):
        # vocabSize containes three tags: [START], [END] and [UNK]
        self.vocabSize = vocabSize

    def buildModel(self, trainData):
        self.wordMap = self._getWordMap(trainData)
        print "vocabulary built, with vocabulary size {}".format(min(self.vocabSize, len(self.wordMap)))
        return self

    def buildLookupTabel(self):
        self.wordIndex = {"SOS": 0,  "EOS": 1}
        counter = 2
        for key in self.wordMap:
            if (key == "SOS" or key == "EOS"):
                continue
            self.wordIndex[key] = counter
            counter += 1
        # build wordvec
        print "build initial lookup table"
        return self

    def data4NN(self, Textaddr, batch):
        # batch: is the size of batch
        # grams: is the input data, text file
        data = []
        if Textaddr is None:
            return None
        isFirst = 1
        mem = None
        with open(Textaddr, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.strip().split(" ")
                for i in range(len(line)):
                    if line[i] not in self.wordMap:
                        line[i] = "UNK"
                if isFirst:
                    isFirst = 0
                    mem = line
                    continue
                data.append((mem, line))
                mem = line
        return WordEmoManager(data, self.wordIndex, batch)

    def _getWordMap(self, Textaddr):
        tmpDict = ddict(int)
        endFreq = 0
        startFreq = 0
        with open(Textaddr, 'r', encoding="utf-8") as file:
            for line in file:
                line = line[:-1]
                for i in line.split(" "):
                    i = re.findall("\\w+", i)
                    for w in i:
                        tmpDict[w] += 1
        if (len(tmpDict) > self.vocabSize-1):
            tmpList = sorted([(key, tmpDict[key]) for key in tmpDict], key=lambda x:x[1], reverse=True)
            tmpDict = ddict(int)
            for idx in xrange(self.vocabSize-1):
                tmpDict[tmpList[idx][0]] = tmpList[idx][1]
        tmpDict["UNK"] = 0
        return tmpDict

    def _getNgram(self, Textaddr):
        tmpDict = ddict(int)
        with open(Textaddr, 'r') as file:
            for line in file:
                line = line.strip()
                line = ("START " + line.lower() + ' END').split(" ")
                if len(line) < self.Ngram:
                    continue
                for i in range(len(line)):
                    if line[i] not in self.wordMap:
                        line[i] = "UNK"
                for i in range(len(line)-self.Ngram+1):
                    tmpDict[" ".join(line[i:i+self.Ngram])] += 1
        return tmpDict

    @staticmethod
    def topNGram(Ngrams, topN):
        if topN > len(Ngrams):
            print "error: top-N should be less than or equal to the size of Ngram model"
            return None
        else:
            rst = [(Ngrams[key], key) for key in Ngrams]
            return heapq.nlargest(topN, rst)


class WordEmoManager:
    def __init__(self, data, lookupTable, batch):
        # data, a list[]
        self.rawdata = data
        self.lookupTable = lookupTable
        self.batch = len(data) if batch is None else batch
        self.inverseIdx = None
        self.emoCls = None
        self.emoIdx = None

    def setEmotionCls(self, emoCls):
        self.emoCls = []
        with open(emoCls, 'r') as file:
            self.emoCls = [line.strip() for line in file]
        
    def setEmoIdx(self, emoIdx):
        # emoIdx: a dictionary
        self.emoIdx = emoIdx

    def getBatch(self, shuffled=True):
        # batch
        #print self.batch, "get batch"
        if shuffled:
            random.shuffle(self.rawdata)
        batches = it.izip(xrange(0, len(self.rawdata)+1, self.batch), xrange(self.batch, len(self.rawdata)+1, self.batch))
        for start, end in batches:
            #print start, end
            if self.emoCls and self.emoCls:
                yield (self._Helper(start, end), [self.emoIdx[i] for i in self.emoCls[start:end]])
            else:
                yield self._Helper(start, end)
    """
    def getWordFromIdx(self, idxs):
        # idxs: [batch, ]
        if self.inverseIdx is None:
            self.inverseIdx = {self.lookupTable[key]:key for key in self.lookupTable}
        rst = [self.inverseIdx[idxs[i]] for i in range(idxs.shape[0])]
        return " ".join(rst)
    """


    def getWordFromIdx(self, idxs):
        # idxs: [batch, ]
        if self.inverseIdx is None:
            self.inverseIdx = {self.lookupTable[key]: key for key in self.lookupTable}
        rst = [self.inverseIdx[i] for i in idxs]
        return " ".join(rst)


    def _Helper(self, start, end):
        tmp = self.rawdata[start:end]
        idx1 = []
        idx2 = []
        for pair in tmp:
            a = pair[0]
            b = pair[1]
            #print a==None, b==None, start, end, len(self.rawdata)
            idx1.append([self.lookupTable[w] for w in a])
            idx2.append([self.lookupTable[w] for w in b])
        return (idx1, idx2)

if __name__ == "__main__":
    test = "/media/lyma/entertain/cmu/Semester3/10707DL/subtitileData/tiny.txt"
    demo = DataManager()
    demo.buildModel(test)
    print len(demo.wordMap)
    demo.buildLookupTabel()
    wm = demo.data4NN(test, 1)
    EMOCLS = "/media/lyma/entertain/cmu/Semester3/10707DL/subtitileData/tiny_emotion.txt"
    gen = wm.getBatch()
    for i in gen:
        print len(i[0])
    wm.setEmotionCls(EMOCLS)
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
    wm.setEmoIdx(emoIdx)
    gen = wm.getBatch()
    for i in gen:
        print len(i[0])