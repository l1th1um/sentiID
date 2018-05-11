from __future__ import division
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
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
from collections import Counter
from sentiID import DictSentiment
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import json
import sys
import re

class extractSentiVal(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.mysenti = DictSentiment.DictSentiment()
		
	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		o_data_sv = []
		i = 0
		for tweet in posts:
			print ("sentival-" + str(i))
			senti_str = json.loads(self.mysenti.debug(tweet))
			o_data_sv.append({})
			for word in senti_str:
				if word not in stopwords:
					o_data_sv[i][word + "_pos"] = senti_str[word]["pos"] 
					o_data_sv[i][word + "_neg"] = senti_str[word]["neg"]    
                
			i += 1	
		return o_data_sv


###Preprocess tweets
def processTweet2(tweet):
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

def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
	stopwords = []
	stopwords.append('AT_USER')
	stopwords.append('URL')
	with open(stopWordListFileName) as f:
		for line in f:
			stopwords.append(line.strip())
	return stopwords


def replaceTwoOrMore(s):
	#look for two or more repetitions of character and replace with the character itself
	pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
	return pattern.sub(r"\1", s)


def getFeatureVector(tweet):
	featureVector = []
	#split tweet into words
	words = tweet.split()
	for w in words: 
		#replace two or more with single character
		w = replaceTwoOrMore(w) 
		#strip punctuation
		w = w.strip('\'"?,.')
		#check if the word starts with an alphabet
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
		#ignore if it is a stop word
		if(w in stopwords or val is None):
			continue
		else:
			featureVector.append(w.lower())
	return featureVector


if __name__ == '__main__':
	###load data################################################
	path_data = 'data/'
	###data online shop
	file_train_pos = 'olshop_positive_training.csv'
	file_train_neg = 'olshop_negative_training.csv'
	file_tr_pos = pd.read_csv(path_data + file_train_pos, delimiter='|')
	file_tr_neg = pd.read_csv(path_data + file_train_neg, delimiter='|')
	file_tr_pos["sentiment"] = "positive"
	file_tr_neg["sentiment"] = "negative"
	file_tr_all = [file_tr_pos,file_tr_neg]
	data_tr_ol = pd.concat(file_tr_all,ignore_index=True)
	data_tr_ol['text'] = data_tr_ol['title'] + " " + data_tr_ol['text']

	file_test_pos = 'olshop_positive_testing.csv'
	file_test_neg = 'olshop_negative_testing.csv'
	file_te_pos = pd.read_csv(path_data + file_test_pos, delimiter='|')
	file_te_neg = pd.read_csv(path_data + file_test_neg, delimiter='|')
	file_te_pos["sentiment"] = "positive"
	file_te_neg["sentiment"] = "negative"
	file_te_all = [file_te_pos,file_te_neg]
	data_te_ol = pd.concat(file_te_all,ignore_index=True)
	data_te_ol['text'] = data_te_ol['title'] + " " + data_te_ol['text']
	###concat data onlineshop
	data_ol = pd.concat([data_tr_ol,data_te_ol],ignore_index=True)


	###data twitter
	file_train_tweet = 'tweets_training.csv'
	data_tr_tw = pd.read_csv(path_data + file_train_tweet, delimiter='|',names=['id','text','sentiment'])
	data_tr_tw['sentiment'].replace([1,-1],["positive","negative"],inplace=True)
	file_te_tweet = 'tweets_testing.csv'
	data_te_tw = pd.read_csv(path_data + file_te_tweet, delimiter='|',names=['id','text','sentiment'])
	data_te_tw['sentiment'].replace([1,-1],["positive","negative"],inplace=True)
	###concat data twitter
	data_tw = pd.concat([data_tr_tw,data_te_tw],ignore_index=True)
			
	###data online news from twitter
	file_netral = 'tweets_berita.csv'
	data_netral = pd.read_csv(path_data + file_netral, delimiter='|',names=['id','text','sentiment'])
	
	###combine all data################################################# 
	data_all = pd.concat([data_ol,data_tw,data_netral],ignore_index=True)
	data_all.sample(frac=1)

	stopwords = getStopWordList('stopwords_id.txt')
	x_temp = set()
	X = []
	y = []
	i = 0
	for i in range(len(data_all)):
		tweet = data_all["text"][i]
		sentiment = data_all['sentiment'][i]
		print ("data-" + str(i) + " " + str(sentiment))
		processedTweet = processTweet2(str(tweet))
		featureVector = getFeatureVector(processedTweet)
		newTweet = " ".join(featureVector)
		if newTweet not in x_temp:		#avoid duplicate data
			x_temp.add(newTweet)
			X.append(newTweet)
			y.append(sentiment)  
			i += 1

	x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

	tfidf_vectorizer = TfidfVectorizer()
	tp_vectorizer = CountVectorizer(min_df=5, binary=True)
	
	sv_vectorizer = Pipeline([
		('sv', Pipeline([				#find sentiment value for every sentiment words
			('extract', extractSentiVal()),	
			('dictvect', DictVectorizer(sparse=False))
		]))
	])   
	comb_vectorizer = Pipeline([
		('features', FeatureUnion([
			('tfidf', tfidf_vectorizer),	#find tfidf value
			('tp', tp_vectorizer),			#find term presence
			('sv', sv_vectorizer)
		]))
	])   


	tfidf_train = tfidf_vectorizer.fit_transform(x_train).todense()
	tfidf_test = tfidf_vectorizer.transform(x_test).todense()

	tp_train = tp_vectorizer.fit_transform(x_train)
	tp_test = tp_vectorizer.transform(x_test)

	sv_train = sv_vectorizer.fit_transform(x_train)
	sv_test = sv_vectorizer.transform(x_test)

	c_train = comb_vectorizer.fit_transform(x_train)
	c_test = comb_vectorizer.transform(x_test)


	path_result= "result/"
	file_res = open(path_result + "output_desc.txt","w")

	file_res.write("data training: " + str(len(x_train)) + "\n") 
	file_res.write("data testing: " + str(len(x_test)) + "\n") 
	file_res.write("\n")
	file_res.write("data training pos: " + str(sum(i=="positive" for i in y_train))+ "\n") 
	file_res.write("data training neg: " + str(sum(i=="negative" for i in y_train))+ "\n") 
	file_res.write("data training net: " +  str(sum(i=="neutral" for i in y_train))+ "\n") 
	file_res.write("data testing pos: " + str(sum(i=="positive" for i in y_test))+ "\n") 
	file_res.write("data testing neg: " + str(sum(i=="negative" for i in y_test))+ "\n") 
	file_res.write("data testing net: "+  str(sum(i=="neutral" for i in y_test))+ "\n") 
	file_res.write("feature tfidf: " + str(tfidf_train.shape) + "\n") 
	file_res.write("feature tp: " + str(tp_train.shape) + "\n") 
	file_res.write("feature sv: " + str(sv_train.shape) + "\n") 
	file_res.write("feature comb: " + str(c_train.shape) + "\n") 


	mysenti = DictSentiment.DictSentiment()
	y_base = []
	for tweet in x_test:
		pred = mysenti.predict(tweet).split()[0]
		pred = pred.replace('tif','tive')
		pred = pred.replace('netral','neutral')
		y_base.append(pred) 
	file_res.write("\n")
	file_res.write("Testing (summary) Baseline -> Prec: " + str(precision_score(y_test,y_base,average='weighted')) + " Rec: " + str(recall_score(y_test,y_base,average='weighted')) + " F1-score: " + str(f1_score(y_test,y_base,average='weighted')) + " Accuracy: " + str(accuracy_score(y_test,y_base,normalize=True)) + "\n")

	model1 = Pipeline([('comb',comb_vectorizer),
		('LinSVC',LinearSVC())])
	model2 = Pipeline([('comb',comb_vectorizer),
		('NBM',MultinomialNB())])
	model3 = Pipeline([('comb',comb_vectorizer),
		('LR',LogisticRegression())])
	model4 = Pipeline([('comb',comb_vectorizer),
		('DT',DecisionTreeClassifier())])


	mdl = [
		model1,
		model2,
		model3,
		model4	
	]

	estimators = []
	i = 0
	for i in range(len(mdl)):
		estimators.append(('mdl-'+ str(i),mdl[i]))
	mdl_en = VotingClassifier(estimators, voting='hard')	#ensemble
	mdl.append(mdl_en)

	
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	for clf, label in zip(mdl, ['LinearSVC', 'Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Ensemble Hard']):
		print ("processing model " + label) 
		kfold_train = cross_val_score(clf, x_train, y_train, cv=kfold)
		
		clf.fit(x_train,y_train)
		result= clf.predict(x_test)

		file_res.write("Accuracy model " + label + ":" + str(kfold_train.mean()*100) + "\n")
		file_res.write("Accuracy testing model " + label + " -> Prec: " + str(precision_score(y_test,result,average='weighted')) + " Rec: " + str(recall_score(y_test,result,average='weighted')) + " F1-score: " +  str(f1_score(y_test,result,average='weighted')) + " Accuracy: " + str(accuracy_score(y_test,result,normalize=True)) + "\n\n")
		file_out = open(path_result + label + ".csv","w")
		i= 0
		for i in range(len(y_test)):
			file_out.write(x_test[i] + "|" + str(y_test[i]) + "|" + str(y_base[i]) + "|" + str(result[i]) + "\n")
		file_out.close()

	print ("Finish.....")


	