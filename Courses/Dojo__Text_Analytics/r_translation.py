
import os
import re
import heapq
from pprint import pprint
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

workdir = dirname(__file__)
os.chdir(workdir)


# Data Exploratory Analysis 
# --------------------------------------------------

df = pd.read_csv('spam.csv')    
df.head(2)

# Remove unnecesary columns
df = df.iloc[:,:2]
df.columns = ['Label','Text']

# Convert class to a Label (Factor in R)
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].astype("category")
categories = pd.Categorical(df['Label']).categories
# Missing Values
df.info()

# Class balanceness
df['Label'].hist()
print(df.Label.value_counts().values / len(df))

# Length of the emails
df['Length'] = df['Text'].apply(lambda x: len(x))
df['Length'].describe()

# Distribution of Lenght of Texts based on their labels
fig = plt.figure()
ax = plt.subplot(111)
for cat in categories:
    ax.hist(df.loc[df['Label'] == cat, 'Length'],bins=50)
plt.show()


# Prepare Data for Modeling 
# ---------------------------------------------------

X = np.arange(10)
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

sampler1 = StratifiedKFold(n_splits=3)
sampler2 = StratifiedShuffleSplit(n_splits=3,test_size=0.4)

print('Stratified-K-Fold')
for t_id, v_id in sampler1.split(X=X,y=y):
    print(t_id,v_id)

print('Stratified-Shuffle-Split')
for t_id, v_id in sampler2.split(X=X,y=y):
    print(t_id,v_id)



X, y = df['Text'].to_list(), df['Text'].to_list()
