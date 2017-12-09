import re
import codecs
import os

MAXLENGTH = 30
START = "SOS "#.encode("utf-8")
END = " EOS\n"#.encode("utf-8")

def removeUselessCharacter_file(pathOfFile):
	isFirst = 0
	with open(pathOfFile, 'r') as file, open("test.txt", 'a') as out:
		mem = file.readline()
		#mem = mem.encode("utf-8")
		mem = removePrimes(mem[:-1]).lower()
		if (len(mem) > MAXLENGTH):
			mem = mem[:MAXLENGTH]
		men = getMaxLength(mem)
		mem = START + mem + END
		for line in file:
			#line = line.encode("utf-8")
			line = line[:-1]
			# print fmtO.format(line)
			line = removePrimes(line).lower()
			line = getMaxLength(line)
			line = START + line + END
			out.write(mem)
			out.write(line)
			mem = line

def removeUselessCharacter_Sent(s):
	removeHeadHyphen = re.compile("^-\\s")
	jointAbbr = re.compile("\'\\s")
	s = re.sub(jointAbbr, "\'", s)
	s = re.sub(removeHeadHyphen, '', s)
	return s

def getMaxLength(s):
	p = re.compile("\\w+")
	words = re.finditer(p, s)
	idx = 0
	last = -1
	for word in words:
		idx += 1
		if idx == MAXLENGTH:
			last = word.end()
	if last != -1:
		s = s[:last]
	return s



def removePrimes(s):
	removePrime = re.compile("\'\\s")
	removeHeadHyphen = re.compile("^-\\s")
	return re.sub(removeHeadHyphen, "", re.sub(removePrime, ' ', s))

if __name__ == "__main__":
	test = "/media/lyma/entertain/cmu/Semester3/10707DL/OpenSubtitles_Content"
	#removeUselessCharacter_file(test)
	fileList = os.listdir(test)
	L = len(fileList)-1
	for i in range(2):
		f = test + "/" + fileList[L-i]
		print f
		removeUselessCharacter_file(f)