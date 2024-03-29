# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 00:43:43 2016

@author: ttw
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


df_in = pd.read_csv(r'datasets/train_in.csv',header=None, skiprows=1, usecols=[1])
data_in = [val for sublist in df_in.values.tolist() for val in sublist]
            
df_out = pd.read_csv(r'datasets/train_out.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in df_out.values.tolist() for val in sublist]
            
X_train, X_test, y_train, y_test = train_test_split(data_in,data_out, random_state=10)            
             
tfidf_vectorizer = TfidfVectorizer(max_features=15000) 
train_matrix = tfidf_vectorizer.fit_transform(X_train)
test_matrix = tfidf_vectorizer.transform(X_test)

#sss = StratifiedShuffleSplit(n_splits=5,random_state=12)

logreg = linear_model.LogisticRegression(solver='sag',multi_class='multinomial',n_jobs=-1)
logreg.fit(train_matrix, y_train)
print logreg.score(test_matrix, y_test)

