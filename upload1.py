# -*- coding: utf-8 -*-
"""
Created on Sun Oct 09 00:23:28 2016

@author: ttw
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, preprocessing

df_in = pd.read_csv(r'datasets/train_in.csv',header=None, skiprows=1, usecols=[1])
X_train = [val for sublist in df_in.values.tolist() for val in sublist]
            
df_out = pd.read_csv(r'datasets/train_out.csv',header=None, skiprows=1, usecols=[1])
Y_train = [val for sublist in df_out.values.tolist() for val in sublist]
            
df_in = pd.read_csv(r'datasets/test_in.csv',header=None, skiprows=1, usecols=[1])
X_test = [val for sublist in df_in.values.tolist() for val in sublist]            
            
le = preprocessing.LabelEncoder()
Y_train = le.fit_transform(Y_train)            

tfidf_vectorizer = TfidfVectorizer(max_features=40000, ngram_range=(1,2), max_df=0.9) 
train_matrix = tfidf_vectorizer.fit_transform(X_train)
test_matrix = tfidf_vectorizer.transform(X_test)

logreg = linear_model.LogisticRegression(C=1.7, solver='sag',multi_class='multinomial', max_iter=200,n_jobs=-1 )
logreg.fit(train_matrix, Y_train)
Y_pred = logreg.predict(test_matrix)
Y_pred = le.inverse_transform(Y_pred)

my_df = pd.DataFrame(Y_pred)
my_df.columns=['Category']
my_df.to_csv(r'datasets/test_out.csv', index_label="id")