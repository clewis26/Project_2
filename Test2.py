# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 00:03:16 2016

@author: ttw
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split

df_in = pd.read_csv(r'datasets/train_in.csv',header=None, skiprows=1, usecols=[1])
data_in = [val for sublist in df_in.values.tolist() for val in sublist]
            
df_out = pd.read_csv(r'datasets/train_out.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in df_out.values.tolist() for val in sublist]
            
le = preprocessing.LabelEncoder()
data_out = le.fit_transform(data_out)            

X_train, X_test, y_train, y_test = train_test_split(data_in,data_out)            

tfidf_vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), max_df=0.9) 
train_matrix = tfidf_vectorizer.fit_transform(X_train)
test_matrix = tfidf_vectorizer.transform(X_test)

logreg = linear_model.LogisticRegression(C=1.7, solver='sag',multi_class='multinomial', max_iter=200,n_jobs=-1 )
logreg.fit(train_matrix, y_train)
print logreg.score(test_matrix, y_test)
