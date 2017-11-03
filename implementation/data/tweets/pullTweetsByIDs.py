import tweepy
import re
import sys

consumer_key = "gHBbHLPeLeVtJnedFY46j2VKc"
consumer_secret = "ex194ft8haPbEsAaPcOPdJEFAh36NzYMWQAFNbAAKQWLDyaitk"
access_token = "924404539381420032-w2fqRxrlyK53xrBGi0Y7WSgZiLOkn6Z"
access_token_secret = "9zTfFEleLmJoNCOA5H5eCLdZv02X1QIkrBmru2G3L6f23"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def getTweets(api, _input, outDir, name):
	fmt = "{}\t{}\n"
	fTweets = open(outDir + "/" + name + "_Tweets.txt", 'w')
	fIDMoods = open(outDir + "/" + name + "_IdMoods.txt", 'w')
	fLog = open(outDir + "/" + name + "_error.log", 'w')
	errFmt = "id: {}\t err: {}\n"
	ts = 1000
	with open(_input, 'r') as file:
		c = 0
		for line in file:
			c += 1
			if c %ts == 0:
				print c, "msg ..  #_#"
			tmp = re.findall("\\w+", line)
			try:
				tweet = api.get_status(tmp[0])
			except tweepy.TweepError, err:
				fLog.write(errFmt.format(tmp[0], err))
			else:
				fTweets.write(re.sub(r"\n", ' ', tweet.text.encode("utf-8")) + "\n")
				fIDMoods.write(fmt.format(tmp[0], tmp[1]))	
	print "Done! lol"			

if __name__ == "__main__":
	# sys.argv[1]: set the input data, e.g. /someDirectory/train_1.txt
	# sys.argv[2]: set the output Directory
	# sys.argv[3]: set tht name for file saving
	assert len(sys.argv) == 4
	getTweets(api, sys.argv[1], sys.argv[2], sys.argv[3])
	#print re.sub(r"\n", '', api.get_status("142488672607019009").text)