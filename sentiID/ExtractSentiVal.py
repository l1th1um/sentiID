from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
from sentiID import DictSentiment
import json
import sys
import re
import pkg_resources

class ExtractSentiVal(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.file_sw = pkg_resources.resource_filename('sentiID','data/stopwords_id.txt')
		self.mysenti = DictSentiment()
		self.stopwords = self.getStopWordList()
		
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		o_data_sv = []
		i = 0
		for tweet in posts:
			senti_str = json.loads(self.mysenti.debug(tweet))
			o_data_sv.append({})
			for word in senti_str:
				if word not in self.stopwords:
					o_data_sv[i][word + "_pos"] = senti_str[word]["pos"] 
					o_data_sv[i][word + "_neg"] = senti_str[word]["neg"]    
                
			i += 1	
		return o_data_sv

	def getStopWordList(self):
	    #read the stopwords file and build a list
		stopwords = []
		stopwords.append('AT_USER')
		stopwords.append('URL')
		with open(self.file_sw) as f:
			for line in f:
				stopwords.append(line.strip())
		return stopwords