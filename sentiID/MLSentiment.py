from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sentiID import DictSentiment
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import pandas as pd
import numpy as np
import json
import sys
import re
import pickle
import pkg_resources


class MLSentiment():

	def __init__(self):
		self.file_mdl = pkg_resources.resource_filename('sentiID','data/ensemble_model.pk')
		self.file_sw = pkg_resources.resource_filename('sentiID','data/stopwords_id.txt')
		self.file_emo = pkg_resources.resource_filename('sentiID','data/sentiment_emotions.txt')


	###Preprocess tweets
	def processTweet2(self,tweet):
	     #Convert to lower case
		tweet = tweet.lower()
		#Convert www.* or https?://* to URL
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
		#Convert @username to AT_USER
		tweet = re.sub('@[^\s]+','AT_USER',tweet)
		#Remove additional white spaces
		tweet = re.sub('[\s]+', ' ', tweet)
		#Replace #word with word
		tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
		#trim
		tweet = tweet.strip('\'"')

		return tweet    

	def getStopWordList(self):
	    #read the stopwords file and build a list
		stopwords = []
		stopwords.append('AT_USER')
		stopwords.append('URL')
		with open(self.file_sw) as f:
			for line in f:
				stopwords.append(line.strip())
		return stopwords


	def replaceTwoOrMore(self,s):
		#look for two or more repetitions of character and replace with the character itself
		pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
		return pattern.sub(r"\1", s)


	def getFeatureVector(self, tweet, stopwords):
		#Emoticons
		emo = []
		with open(self.file_emo) as f:
			for line in f:
				row = line.split("\t")
				word = row[0]
				emo.append(word)
		token = [re.escape(word) for word in emo]
		regex = '(?:' + "|".join(token) + ')'
		regex = '(' + regex + ')'
		wordlist = re.compile(regex, flags=re.UNICODE)

		featureVector = []
		#split tweet into words
		words = tweet.split()
		for w in words: 
			#replace two or more with single character
			w = self.replaceTwoOrMore(w) 
			#strip punctuation
			w = w.strip('\'"?,.')
			#check for emoticons
			w_emo = wordlist.findall(w)
			if len(w_emo)>0 :
				featureVector.append(str(w_emo))
			else:
				#check if the word starts with an alphabet
				val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
				#ignore if it is a stop word
				if(w in stopwords or val is None):
					continue
				else:
					featureVector.append(w.lower())
		return featureVector

	#Predict the sentiment

	def predict(self,tweet):   
		classifier = pickle.load(open(self.file_mdl, 'rb'))
		processedTweet = self.processTweet2(str(tweet))
		stopwords = self.getStopWordList()
		featureVector = self.getFeatureVector(processedTweet, stopwords)
		newTweet = " ".join(featureVector)
		X = [newTweet]
		if len(newTweet.split())<4:
			mysenti = DictSentiment()
			sentiment = mysenti.predict(newTweet)
			return (sentiment.split()[0])
		else:
			sentiment = classifier.predict(X)
			return (sentiment[0])
