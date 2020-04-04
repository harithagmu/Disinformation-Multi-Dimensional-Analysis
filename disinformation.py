disinformation.py

'''
1. Read data
	LIAR
	BS Detector
	FakenewsNET
2.Preliminary analysis
	Plot Word frequency
	Word cloud

3. Formatting
	Stop words
	Stemming
	Lemmatization
	Tokenization
4. Vector Representations
	Basic
		Bag of Words
		Conti Bag of Words
		TF- IDF representation
		n-grams
	Other pretrained embeddings
		GLOVE
		Word2Vec
		Learn the language model from training data

5. Classification architechtures Training
	Baseline
		Naive Bayes
		Logistic Regression
	Advanced
		Deep Neural Network
		LSTM
		Bidirectional LSTMs
		Others based on performance
6. Performance Evaluation
	Parameter Tuning for each of the models
		Validation accuracy
	Test Accuracy 
	Confusion Matrix


Questions to consider?
	Can  we combine the data
	Check SBPBRiMs challenge page for more information
'''



import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing
import time
from sklearn.model_selection import StratifiedKFold
nl.download('punkt')


start_time = time.time()

def clean(df):

    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext

    def removeat(text):
        atlist=[]
        for word in text:
            pattern = re.compile('^@')
            if re.match(pattern,word):
                #cleantext1 = re.sub(pattern, word[1:], word)
                atlist.append(word[1:])
            else:
                atlist.append(word)
        return atlist

    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist

    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))


    tokentweet=[]
    df.text= cleantweet
    for doc in df.text:
        tokentweet.append(TweetTokenizer().tokenize(doc))
    df.text= tokentweet

    removeattweet=[]
    for doc in df.text:
        removeattweet.append(removeat(doc))
    df.text =removeattweet

    lowertweet=[]
    for doc in df.text:
        lowertweet.append(tolower(doc))
    df.text = lowertweet

    tweets=[]
    for x in df.text:
        tweet = ''
        for word in x:
            tweet += word+' '
        tweets.append(word_tokenize(tweet))
    df.text= tweets

    #stemming
    stemtweets=[]
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    #ps= PorterStemmer()
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    df_unstemmed = pd.DataFrame()
    df_unstemmed['text'] = df['text']
    df_unstemmed['Sentiment'] = df['Sentiment']
    df_stemmed = pd.DataFrame()
    df_stemmed['text'] = df['stemmed']
    df_stemmed['Sentiment'] = df['Sentiment']
   
    return df_stemmed,df_unstemmed

def main():
    import os
    os.chdir("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL")
    # print command line arguments
    train= get_data("tweet\\train") #get_data(sys.argv[1])
    test= get_data("tweet\\test")  #get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    

    print("creating the vocabulary for stemmed and unstemmed data")
    Vocab_stem = createvocab(clean_train_stem)
    Vocab_nostem = createvocab(clean_train_nostem)
    print("vocabulary created")
    print("Stemmed vocabulary length=",len(Vocab_stem))
    print("No stem vocabulary length=",len(Vocab_nostem))
    
    print("*********TRAINING************")
    print("creating positive bigrams")
    
    train_stem_pos_bigram= pos_sample_bigrammer(clean_train_stem)
    train_nostem_pos_bigram= pos_sample_bigrammer(clean_train_nostem)
    
    print("positive samples for training created")
    print("No of stem pos bigrams=" ,len(train_stem_pos_bigram))
    print("No of no stem pos bigrams=", len(train_nostem_pos_bigram))
    

    print("creating negative sample bigrams")
    train_stem_neg_bigram = neg_sample_bigrammer(train_stem_pos_bigram, Vocab_stem,2)
    train_nostem_neg_bigram = neg_sample_bigrammer(train_nostem_pos_bigram, Vocab_nostem,2)
    print("negative samples created")
    
    print("No of stem neg bigrams=" ,len(train_stem_neg_bigram))
    print("No of no stem neg bigrams=", len(train_nostem_neg_bigram))
 

    #create a training dataframe with positive and negative samples and adding labels(1,0)  for them
    train_stem_data = train_stem_pos_bigram + train_stem_neg_bigram
    train_nostem_data= train_nostem_pos_bigram + train_nostem_neg_bigram
    
    #nostem_labels =[0]*len(train_nostem_pos_bigram) + [1]*len(train_nostem_neg_bigram)
    #stem_labels= [0]*len(train_stem_pos_bigram)+ [1]*len(train_stem_neg_bigram)

    print("--- %s seconds ---" %(time.time() - start_time))
    print("train data is ready for stem and no stem ")

    print("Creating Language model")
    stem_model =  Languagemodel(train_stem_data, Vocab_stem, 10)
    Languagemodel(train_nostem_data, Vocab_stem, 10)
    
    print("*********TESTING************")
    print("creating positive bigrams")
    
    test_stem_pos_bigram= pos_sample_bigrammer(clean_test_stem)
    test_nostem_pos_bigram= pos_sample_bigrammer(clean_test_nostem)
    
    print("positive samples for testing created")
    print("No of no stem pos bigrams=", len(test_stem_pos_bigram))
    print("No of stem pos bigrams=" ,len(test_nostem_pos_bigram))
    
    print("creating the vocabulary for stemmed and unstemmed Test data")
    Vocab_stem_test = createvocab(clean_test_stem)
    Vocab_nostem_test = createvocab(clean_test_nostem)
    
    print("creating negative sample bigrams")
    test_stem_neg_bigram = neg_sample_bigrammer(test_stem_pos_bigram, Vocab_stem_test,2)
    test_nostem_neg_bigram = neg_sample_bigrammer(test_nostem_pos_bigram, Vocab_nostem_test,2)
    print("negative samples created")
    
    print("No of no stem neg bigrams=", len(test_nostem_neg_bigram))
    print("No of stem neg bigrams=" ,len(test_stem_neg_bigram))

    #Create labels
    test_stem_data = test_stem_pos_bigram + test_stem_neg_bigram
    test_nostem_data= test_nostem_pos_bigram + test_nostem_neg_bigram
    
    #Testing
    Testing(stem_model, Vocab_stem_test, test_stem_data)
    Testing(stem_model, Vocab_nostem_test, test_nostem_data)
