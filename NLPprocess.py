import pandas as pd 
import numpy as np 

import csv

#instal tweepy - pip install tweepy
import tweepy
from tweepy import OAuthHandler

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import re
import pickle
import matplotlib.pyplot as plt

import json
#import stopword
from stop_words import get_stop_words

#import sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#import heapq
import heapq

#import CRFTagger
from nltk.tag import CRFTagger

# import wordnet
from nltk.corpus import wordnet

#normalization
from modulenorm.modNormalize import normalize
from modulenorm.modTokenizing import tokenize

# 00 Crawling data from twiter
def get_data_from_twitter(search_word):
	consumer_key = 'AI0fX1nKS9q5Xz2J99pF8ZPp4'
	consumer_secret = 'pqS9hSUMEhIgbYop22pnZ9A3dC5flRLDc28sfciyLDCwUu3NMx'
	access_token = '972438193709580288-qPxYFhdbjjD8uXaigrNG7kaSNx3aBzA'
	access_secret = 'xiVbGQPLy6VNuwIotIn2Xpjx08DY9CSfB8AHgjcqLYDQd'

	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
	args = [search_word]
	api = tweepy.API(auth,timeout=100)

	# Fetching the tweets
	list_tweet = []

	query = args[0] 
	if len(args) == 1:
		for status in tweepy.Cursor(api.search, q = query + " -filter:retweets", lang = 'id', result_type = 'recent').items(25): #-filter:retweet searching for no retweets , since = '2018-08-23'
			list_tweet.append(status.text)
	
	return list_tweet

# 01 Case holding
def function_case_holding(list_tweet):
	caseHolding_tweets = []
	for n in range(len(list_tweet)):
		caseHolding = list_tweet[n].lower()
		caseHolding_tweets.append(caseHolding)
	return caseHolding_tweets

# 02 cleansing
def function_cleansing_tweet(caseHolding_tweets, list_tweet):
	cleansing_tweet = caseHolding_tweets
	new_cleansing_tweets = []
	for n in range(len(cleansing_tweet)):
		cleansing_tweet[n] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', cleansing_tweet[n]) # remove link
		cleansing_tweet[n] = re.sub(r"\W", " ", cleansing_tweet[n]) # Matches any character which is not a word character
		cleansing_tweet[n] = re.sub(r"\d", " ", cleansing_tweet[n]) # Matches any Unicode decimal digit 
		cleansing_tweet[n] = re.sub(r"\\s+[a-z]\s+", " ", cleansing_tweet[n], flags=re.I)
		cleansing_tweet[n] = re.sub(r"\s", " ", cleansing_tweet[n])
		cleansing_tweet[n] = re.sub(r"^\s", " ", cleansing_tweet[n])
		cleansing_tweet[n] = re.sub(r"\s$", " ", cleansing_tweet[n])
		cleansing_tweet[n] = cleansing_tweet[n].strip()

		cleansing_tweets = [cleansing_tweet[n], list_tweet[n]]
		new_cleansing_tweets.append(cleansing_tweets)
	return new_cleansing_tweets

# 03 Normalization
def function_normalization(new_cleansing_tweets):
	new_normalization = []
	for n in range(len(new_cleansing_tweets)):
		result = [new_cleansing_tweets[n][0]]
		for row in result:
			text = row.encode("utf-8")
			text_decode = str(text.decode("utf-8"))
			usenorm = normalize()
			text_norm = usenorm.enterNormalize(text_decode) # normalisasi enter, 1 revw 1 baris
			text_norm = usenorm.lowerNormalize(text_norm) # normalisasi huruf besar ke kecil
			text_norm = usenorm.repeatcharNormalize(text_norm) # normalisasi titik yang berulang
			text_norm = usenorm.linkNormalize(text_norm) # normalisasi link dalam text
			text_norm = usenorm.spacecharNormalize(text_norm) # normalisasi spasi karakter
			text_norm = usenorm.ellipsisNormalize(text_norm) # normalisasi elepsis (…)

			tok = tokenize() # panggil modul tokenisasi
			text_norm = tok.WordTokenize(text_norm) # pisah tiap kata pada kalimat

			text_norm = usenorm.spellNormalize(text_norm) # cek spell dari kata perkata
			text_norm = usenorm.wordcNormalize(text_norm,2) # menyambung kata (malam-malam) (param: textlist, jmlh_loop)

			text_norm = ' '.join(text_norm) # menggabung kalimat tokenize dengan separate spasi
		normalization = [text_norm, new_cleansing_tweets[n][1]]
		new_normalization.append(normalization)
	return new_normalization

# 04 negation tweet
def function_negation_tweet(new_normalization):
	new_negation_tweet = []
	for n in range(len(new_normalization)):
		words = nltk.word_tokenize(new_normalization[n][0])
		new_words = []

		temp_word = ""
		for word in words:
			antonyms = []
			if word in ["tidak", "jangan", "bukan", "gak"]:
				if word == "tidak":
					temp_word = "tidak"
				elif word == "jangan":
					temp_word = "jangan"
				elif word == "bukan":
					temp_word = "bukan"
				else:
					temp_word = "gak"
			elif temp_word in ["tidak", "jangan", "bukan", "gak"]:
				for syn in wordnet.synsets(word):
					for s in syn.lemmas():
						for a in s.antonyms():
							antonyms.append(a.name())
				if len(antonyms) >= 1:
					word = antonyms[0]
				else:
					word = temp_word + word
				temp_word = ""
			if word not in ["tidak", "jangan", "bukan", "gak"]:        
				new_words.append(word)

		sentence = ' '.join(new_words)
		negation_tweet = [sentence, new_normalization[n][1]]
		new_negation_tweet.append(negation_tweet)
	return new_negation_tweet

# 05 Tokenizing Tweet
def function_tokenize_tweet(new_negation_tweet):
	new_tokenize_tweets = [] # variabel for storage 
	for n in range(len(new_negation_tweet)):
		# Tokenizing words
		words = nltk.word_tokenize(new_negation_tweet[n][0])
		tokenize_tweet = [words, new_negation_tweet[n][1]]
		new_tokenize_tweets.append(tokenize_tweet)
	return new_tokenize_tweets

# 06 Stopword Tweet
def funtion_stopword_tweet(new_tokenize_tweets):
	stop_words = get_stop_words('id')
	stop_words = get_stop_words('indonesian')

	new_stopwords_tweets = []
	for n in range(len(new_tokenize_tweets)):
		stopword_tweet = []
		for word in new_tokenize_tweets[n][0]:
			if word not in stop_words:
				stopword_tweet.append(word)
			else:
				pass
		stopwords_tweet = [stopword_tweet, new_tokenize_tweets[n][1]]
		new_stopwords_tweets.append(stopwords_tweet)
	return new_stopwords_tweets

# 07 Part Of Speech Tagged 
def function_pos_tagging(new_stopwords_tweets):	
	ct = CRFTagger()
	ct.set_model_file('data/all_indo_man_tag_corpus_model.crf.tagger')
	new_pos_tweets = []
	for n in range(len(new_stopwords_tweets)):
		pos_tweet_word = [new_stopwords_tweets[n][0]]
		pos_tweet_words = ct.tag_sents(pos_tweet_word)
		pos_tweet = [pos_tweet_words, new_stopwords_tweets[n][1]]
		new_pos_tweets.append(pos_tweet)

	new_features_tweets = []
	for n in range(len(new_pos_tweets)):
		pos_tweets_data = new_pos_tweets[n][0][0]
		features = []
		for tokenTag in pos_tweets_data:
			token, tag = tokenTag
			access = ['NN', 'JJ', 'RB', 'VBD']
			if tag in access:
				features.append(token)
			else:
				pass

		if features:
			features_tweets = [features, new_pos_tweets[n][1]]
			new_features_tweets.append(features_tweets)
		else:
			pass
	return new_features_tweets

# 08 stemming
def function_stemmer_tweet(new_features_tweets):
	# create stemmer
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	#stem
	new_stemming_tweets = []
	for n in range(len(new_features_tweets)):
		stemming_output = []
		for m in range(len(new_features_tweets[n][0])):
			output = stemmer.stem(new_features_tweets[n][0][m])
			stemming_output.append(output)

		stemming_tweet= [stemming_output, new_features_tweets[n][1]]
		new_stemming_tweets.append(stemming_tweet)
	return new_stemming_tweets

# 09 lexicon based use wornet bahasa
def funtion_lexicon_based(new_stemming_tweets):
	open_file = open("data/negative.txt", "r")
	stri= ""    #create empty string to manipulate data 
	for line in open_file:
		stri+=line 
	negative_word = stri.split()    #split the string and convert it into list

	open_file = open("data/positive.txt", "r")
	stri= ""    #create empty string to manipulate data
	for line in open_file:
		stri+=line 
	positive_word = stri.split()    #split the string and convert it into list
	
	lexicon_tweets  = []
	for n in range(len(new_stemming_tweets)):
		label_tweet_word = {}
		for word in new_stemming_tweets[n][0]:
			if word in positive_word:
				# words = word + " : True"
				label_tweet_word['%s' % word] = (True)
			elif word in negative_word:
				# words = word + " : False"
				# label_tweet_word.append(words)
				label_tweet_word['%s' % word] = (False)
			else:
				pass

		if label_tweet_word:
			label_tweets = [label_tweet_word, new_stemming_tweets[n][1]]
			lexicon_tweets.append(label_tweets)
		else:
			pass
	return lexicon_tweets

with open("trainingdata.txt", "rb") as fp:   # Unpickling
		NBClassifier = pickle.load(fp)


# 10 naive bayes testing
def function_naive_bayes(lexicon_tweets):
	data = []
	for n in range(len(lexicon_tweets)):
		data_tweets = []    
		dist = NBClassifier.prob_classify(lexicon_tweets[n][0])
		sentiment = NBClassifier.classify(lexicon_tweets[n][0])
		for label in dist.samples():  
			data_tweets.append(dist.prob(label))
		dist_sentiment = [data_tweets, sentiment]
		dist_sentiments = [lexicon_tweets[n][1], dist_sentiment]
		data.append(dist_sentiments)
	return data

def function_information_classifier(function_naive_bayes):
	positive = 0
	negative = 0
	for n in range(len(function_naive_bayes)):
		if function_naive_bayes[n][1][1] == "|positive|":
			positive += 1
		elif function_naive_bayes[n][1][1] == "|negative|":
			negative += 1

	total_tweet = len(function_naive_bayes)

	positif = round(positive / total_tweet * 100)
	negatif = round(negative / total_tweet * 100)
	information = [positif, negatif]

	return information

