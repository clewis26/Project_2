# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 00:43:43 2016

@author: ttw
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df_in = pd.read_csv(r'datasets/train_in.csv',header=None, skiprows=1, usecols=[1])
train_in = [val for sublist in df_in.values.tolist() for val in sublist]
            
df_out = pd.read_csv(r'datasets/train_out.csv',header=None, skiprows=1, usecols=[1])
train_out = [val for sublist in df_out.values.tolist() for val in sublist]

tfidf_vectorizer = TfidfVectorizer(norm='l2') 
tfidf_matrix = tfidf_vectorizer.fit_transform(train_in)


