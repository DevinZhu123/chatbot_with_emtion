import re
from collections import defaultdict as ddict
import sys
from HTMLParser import HTMLParser # for python 2.6 ~ 2.7
import codecs


def extractTag():
	p = re.compile("#\\w+")
	dS = ddict(list)
	d = ddict(int)
	for i in sys.stdin:
		i = i[:-1]
		tmp = re.findall(p, i)
		if tmp == None:
			continue
		else:
			for key in tmp:
				key = re.search("\\S+", key).group()
				d[key] += 1
				dS[key].append(i)
	print "size of the tag dictionary: ", len(d)
	displayDict(dS)

def displayDict(d):
	tmp = sorted([(key,d[key]) for key in d], key=lambda x:x[1], reverse=True)
	for i in tmp:
		print i[0], ": ", i[1]

def dataClean(addr, out):
	p = re.compile("#\\w+")
	fmtO = "{} ori: "
	fmtP = "{} re:  "
	outF = codecs.open(out, 'w', 'utf-8')
	with codecs.open(addr, 'r', 'utf-8') as file:
		c = 0
		zernLine = 0
		for i in file:
			c += 1
			i = i[:-1]
			#print fmtO.format(i)
			# remove tag at tail
			tmp = re.finditer(p, i)
			matchs = sorted([(j.start(), j.end()-1) for j in tmp], key=lambda (s, e):s, reverse=True)
			if len(matchs) == 1 and matchs[0][0] == 0 and matchs[0][1] == len(i)-1:
				c -= 1
				continue
			#for loc in matchs:
			#	if loc[1] == len(i)-1:
			#		i = i[:loc[0]].strip()
			if len(i) == 0:
				c -= 1
				continue
			i = i.strip()
			i = removeHashTag(i)
			i = removeAtTag(i)
			i = htmlEntity(i)
			i = removeEmoji(i)
			outF.write(i + '\n')
			#outF.write(fmtO.format(c) + i + "\n")
			#outF.write(fmtP.format(c) + i + "\n")
	print c, zernLine
	outF.close()

def removeHashTag(s):
	return re.sub("#(?=\\w+)", '', s)

def removeAtTag(s):
	return re.sub("@(?=\\w+)", '', s)
def removeEmoji(s):
	emoji_pattern = re.compile(u'['
							    u'\U0001F300-\U0001F5FF'
							    u'\U0001F600-\U0001F64F'
							    u'\U0001F680-\U0001F6FF'
							    u'\u2600-\u26FF\u2700-\u27BF]+', 
							    re.UNICODE)
	return re.sub(emoji_pattern, " ", s)

def htmlEntity(s):
	p = re.compile("&\\w+;")
	tmp = re.findall(p, s)
	if len(tmp) == 0:
		return s
	entities = set(tmp)
	h = HTMLParser()
	for e in entities:
		p = re.compile(e)
		s = re.sub(p, h.unescape(e), s)
	return s

def extractHtmlTag():
	p = re.compile("&\\w+;")
	fmtO = "ori: {}"
	fmtP = "tag: {}"
	d = ddict(int)
	for i in sys.stdin:
		i = i[:-1]
		tmp = re.findall(p, i)
		for j in tmp:
			d[j] += 1
		if len(tmp) > 0:
			print fmtO.format(i)
			print fmtP.format(tmp)
	#displayDict(d)


if __name__ == "__main__":
	assert len(sys.argv) == 3
	dataClean(sys.argv[1], sys.argv[2])