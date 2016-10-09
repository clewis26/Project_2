# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 00:43:43 2016

@author: ttw
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model, preprocessing, pipeline
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV
from sklearn.decomposition import TruncatedSVD

df_in = pd.read_csv(r'datasets/train_in.csv',header=None, skiprows=1, usecols=[1])
data_in = [val for sublist in df_in.values.tolist() for val in sublist]
            
df_out = pd.read_csv(r'datasets/train_out.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in df_out.values.tolist() for val in sublist]
            
le = preprocessing.LabelEncoder()
data_out = le.fit_transform(data_out)            

X_train, X_test, y_train, y_test = train_test_split(data_in,data_out, random_state=10)            

tfidf_vectorizer = TfidfVectorizer(max_features=15000) 
train_matrix = tfidf_vectorizer.fit_transform(X_train)
test_matrix = tfidf_vectorizer.transform(X_test)

svd = TruncatedSVD()
normalizer = preprocessing.Normalizer(copy=False)
lsa = pipeline.make_pipeline(svd, normalizer)
train_matrix = lsa.fit_transform(train_matrix)
test_matrix = lsa.transform(test_matrix)

sss = StratifiedShuffleSplit(n_splits=3,random_state=12)

param_grid = {'Cs': [1, 10, 100, 1000] }

clf = GridSearchCV(linear_model.LogisticRegressionCV(cv=sss, solver='sag',multi_class='multinomial', max_iter=200 ), param_grid)
clf.fit(train_matrix, y_train)
print clf.score(test_matrix, y_test)

