import xml.sax
import os
import re
import gzip, shutil
import codecs
import sys
import subtitleProcess as sp

class MovieSubHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.time = ""
        self.w = ""
        self.sentence = ""
        self.contents = []

    def storeContent(self, outPath):
        with codecs.open(outPath, 'w', 'utf-8') as file:
            for s in self.contents:
                # remove enter
                s = re.sub("\n", ' ', s)
                file.write(s + "\n")
        self.contents = []

    def allSubtileInOne(self, fh):
        for s in self.contents:
            s = re.sub("\n", ' ', s)
            fh.write(s + "\n")
        self.contents = []

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "s":
            if self.sentence == "":
                pass 
            else:
                self.contents.append(sp.removeUselessCharacter_Sent(self.sentence))
            self.sentence = ""

    def endElement(self, tag):
        if self.CurrentData == "w":
            self.sentence += self.w
            self.sentence += " "
        self.CurrentData = ""

    def characters(self, content):
        if self.CurrentData == "time":
            self.time = content[0]
        elif self.CurrentData == 'w':
            self.w = content


def isType(absPath, tp):
    # check whether the file defined by the absolute path is 
    # an XML file
    # file type is based on expension, extracted by regex
    assert not os.path.isdir(absPath), "Input should be an absolute path of a file, not a directory"
    baseName = os.path.basename(absPath)
    p = re.compile("(?<=.)\\w+")
    tags = re.findall(p, baseName)
    return tags[-1].lower() == tp.lower()

class FindFile:
    # DFS to find files
    def __init__(self, ftype=None):
        self.ftype=ftype

    def _checkFtype(self):
        return self.ftype is not None

    def findFiles(self, curPath, ftype=None):
        # ftype, a string
        # pwd, should not be None!!!
        assert self._checkFtype() or ftype is not None, "must determine a file type to track"
        assert curPath is not None, "current path should not be None"
        if not self._checkFtype():
            self.ftype = ftype
        rst = []
        self._search(curPath, rst, isType)
        return rst

    def _search(self, curPath, rst, checkTypeFun):
        ## recursively finds the target files
        # define stop criterion
        isfile = os.path.isfile(curPath)
        if isfile:
            if checkTypeFun(curPath, self.ftype): rst.append(curPath)
        else:
            # is a directory
            items = os.listdir(curPath)
            for item in items:
                self._search(curPath + "/" + item, rst, checkTypeFun)

class Unzip:
    @staticmethod
    def unzip(fileList):
        p = re.compile(".gz")
        for item in fileList:
            if isType(item, 'gz'):
                tmp = re.search(p, item)
                outFile = item[:tmp.start()]
                with gzip.open(item, 'rb') as f_in, open(outFile, 'w') as f_out:
                    shutil.copyfileobj(f_in, f_out)

def testParser():
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = MovieSubHandler()
    parser.setContentHandler(Handler)
    parser.parse("./test.xml")
    Handler.storeContent("./out_test_sentence.txt")

def testUnzip():
    TYPE = 'gz'
    curPath = "/home/lui/CMU/Semester3/10707/proj/OpenSubtitles"
    demo = FindFile(TYPE)
    print "file type: ", demo.ftype
    rst = demo.findFiles(curPath)
    print "number of files:", len(rst)
    print "unzip files: ......"
    Unzip.unzip(rst)
    print "unzip file done!"
    print "check unzip file number"
    TYPE = 'xml'
    curPath = "/home/lui/CMU/Semester3/10707/proj/OpenSubtitles"
    rst = demo.findFiles(curPath, TYPE)
    print "number of files:", len(rst)

def main(curPath, outDir, oneFile=False):
    TYPE = 'xml'
    demo = FindFile(TYPE)
    rst = demo.findFiles(curPath, TYPE)
    print "Number of Files:", len(rst)
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = MovieSubHandler()
    parser.setContentHandler(Handler)
    if oneFile:
        with codecs.open(outDir + "/allSubtiles.txt", 'w', 'utf-8') as file:
            for i in range(len(rst)):
                print "processing file {}".format(i+1)
                file.write(">>>> subtile {}\n".format(i+1))
                parser.parse(rst[i])
                Handler.allSubtileInOne(file)
    else:
        fmt = "/subtile_{}.txt"
        for i in range(len(rst)):
            print "processing file {}".format(i+1)
            parser.parse(rst[i])
            Handler.storeContent(outDir + fmt.format(i+1))
    print "Done !"


if (__name__ == "__main__"):
    main(sys.argv[1], sys.argv[2], True)
    