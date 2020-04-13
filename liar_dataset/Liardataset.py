import os
import sys
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import timeit


from collections import defaultdict
from pathlib import Path

import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
nl.download('stopwords')

import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing

from transformers import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

path= "D:/Spring 2020/Faker/liar_dataset"
stdoutorigin = sys.stdout
sys.stdout = open("D:\\Spring 2020\\Faker\\liar_dataset\\Logfile.txt", "w")

def read_data(file):
    df = pd.read_csv(file, sep='\t')
    colnames = ['file', 'tag', 'text', 'topic', 'speaker', 'speaker_job', 'speaker_state','speaker_party', 'ch_false', 'ch_barelytrue','ch_halftrue','ch_mostlytrue','ch_pantsonfire', 'Spoke_at' ]
    df.columns = colnames
    df.drop('file', axis = 1, inplace = True)
    return df

def cleandataset(data):    
    for col in data.columns:
        if data[col].dtype == np.object:
            data[col].fillna(value='none', inplace=True)
    data.fillna(value = 0, inplace=True)
    data = tokenise(data)
    data.text = removestopwords(data.text)
    return data
    
def tokenise(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    templist= list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t), df.text))
    df.text= templist
    return df

def removestopwords(text):
    x=[]
    for tokens in text:
        x.append([word for word in tokens if word not in stopwords.words('english')])
    return x

def getrep(train_text, val_text, rep='tfidf'):
    def dummy_fun(text):
        return text
    
    if rep == 'binary':
        vectorizer = CountVectorizer(binary = True, analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
    if rep == 'freq': #for freq
        vectorizer = CountVectorizer(binary= False,analyzer='word',tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
    if rep == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None)
    if rep == 'ngramfreq':
        vectorizer= CountVectorizer(binary=False,analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
    if rep == 'ngrambin':
        vectorizer= CountVectorizer(binary=True,analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None,ngram_range=(1,2))
    if rep == 'ngramtfidf':
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun,preprocessor=dummy_fun, token_pattern=None, ngram_range=(1,2))
    
    train_vec = vectorizer.fit_transform(train_text)
    vocab = vectorizer.vocabulary_
    print('length of vocab ', len(vocab))
    df_train = pd.DataFrame(train_vec.todense(), columns = vectorizer.get_feature_names())
    
    val_vec = vectorizer.transform(val_text)
    df_val = pd.DataFrame(val_vec.todense(), columns = vectorizer.get_feature_names())
    
    return df_train, df_val

def getdummies(train, validation):
    columnstoOHE = ['topic', 'speaker',	'speaker_state', 'speaker_party']
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_fit = ohe.fit_transform(train[columnstoOHE])
    train_ohe = pd.DataFrame(ohe_fit.todense(), columns = ohe.get_feature_names())
    ohe_trans = ohe.transform(validation[columnstoOHE])
    val_ohe = pd.DataFrame(ohe_trans.todense(), columns = ohe.get_feature_names())
    return train_ohe, val_ohe

def baseline(X, y_train, X_test, y_test, colnames):
    print('cols ', len(colnames))
    print("Logistic regression")
    #Logistic
    LRclf = LogisticRegression(solver='saga').fit(X[colnames], y_train)
    #coef_dict = {}
    #for coef, feat in zip(LRclf.coef_,colnames):
    #    coef_dict[feat] = coef
    #print(coef_dict) 
    print('--------------------------------------------------------------------')
    print('Labels ', LRclf.classes_)
    for subcoefs in LRclf.coef_:
        sublist = list(zip(subcoefs, colnames))
        print('len of each list ', len(sublist))
        print(sublist)
    print('--------------------------------------------------------------------')
    y_lr_pred= LRclf.predict(X_test[colnames])
    print('accuracy : ', accuracy_score(y_test, y_lr_pred))
    print('f1_score : ', f1_score(y_test, y_lr_pred,average='macro'))
    display_labels = list(np.unique(y_train))
    print(confusion_matrix(y_test, y_lr_pred, display_labels))
      
    print("Naive Bayes Classifier")
    #Naive Bayes
    NBclf = BernoulliNB().fit(X, y_train)
    y_nb_pred= NBclf.predict(X_test)
         
    #Compute accuracy
    print('accuracy : ', accuracy_score(y_test, y_nb_pred))
    print('f1_score : ', f1_score(y_test, y_nb_pred, average='macro'))
    print(confusion_matrix(y_test, y_nb_pred, display_labels))
    return y_lr_pred, y_nb_pred
    
def main():
    start = timeit.default_timer()
    os.chdir(path)
    print("----------------------- START -----------------------", start)
    
    train = read_data('train.tsv')
    validation = read_data('valid.tsv')
    
    print('The shape of train and validation datasets are: ')
    print(train.shape)
    print(validation.shape)
    
    #preprocess
    cleaned_train = cleandataset(train)
    cleaned_validation = cleandataset(validation)
    
    #seperate X and y from the dataset
    y_train = cleaned_train.tag.values
    #ohe = OneHotEncoder(handle_unknown='ignore')
    #y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
    #print(y_train_ohe)
    cleaned_train.drop('tag', axis =1, inplace = True)
    y_test= cleaned_validation.tag.values
    #y_test_ohe =ohe.transform(y_test.reshape(-1, 1))
    cleaned_validation.drop('tag', axis =1, inplace = True)
    
    train_o, val_o = getdummies(cleaned_train, cleaned_validation)    
    print('after one hot encoding train size is :', train_o.shape)
    
    output= pd.DataFrame(list(zip(validation.text, y_test)), columns=["text","truth"])
    
    representation = ['tfidf', 'freq']#, 'binary']
    for rep in representation:    
        print('---------------- Using reprentation : {} ------------------'.format(rep))
        #vectorizations
        train_0, val_0 = getrep(cleaned_train['speaker_job'], cleaned_validation['speaker_job'], rep)
        train_1, val_1 = getrep(cleaned_train['Spoke_at'], cleaned_validation['Spoke_at'], rep)
        train_2, val_2 = getrep(cleaned_train['text'], cleaned_validation['text'], rep)
               
        traindf = pd.concat([train_0, train_1, train_2, train_o], axis =1)
        validationdf = pd.concat([val_0, val_1, val_2, val_o], axis =1)
        
        print('After pre-processing the shape of train and validation datasets are: ')
        print(traindf.shape)
        print(validationdf.shape)
        	
        #Normalize the data
        min_max_scaler = preprocessing.MinMaxScaler()
        X = pd.DataFrame(min_max_scaler.fit_transform(traindf.values), columns = traindf.columns)
        X_test = pd.DataFrame(min_max_scaler.transform(validationdf.values), columns = validationdf.columns)
        
        output[rep+'_lr'], output[rep+'_nb'] = baseline(X, y_train, X_test, y_test, traindf.columns)   
        
    output.to_csv("baselineResults.csv")
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    stdoutorigin.close()

if __name__ == "__main__":
    main()