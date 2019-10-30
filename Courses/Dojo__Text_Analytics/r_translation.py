
import os
import re
import heapq
from pprint import pprint
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

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

# # Class balanceness
# df['Label'].hist()
# print(df.Label.value_counts().values / len(df))

# # Length of the emails
# df['Length'] = df['Text'].apply(lambda x: len(x))
# df['Length'].describe()

# # Distribution of Lenght of Texts based on their labels
# fig = plt.figure()
# ax = plt.subplot(111)
# for cat in categories:
#     ax.hist(df.loc[df['Label'] == cat, 'Length'],bins=50)
# plt.show()


# NLTK Pipeline for text processing
# ---------------------------------

# 1. Lowercase
df['Text_Raw'] = df['Text']
df['Text'] = df['Text'].apply(lambda x: x.lower())

# 2. Tokenization and Remove Punk
from nltk.tokenize import RegexpTokenizer       # Custom Regez Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
df['Text'] = df['Text'].apply(tokenizer.tokenize)
# 2.1 - Alternative moving back to sentences
df['Text1'] = df['Text'].apply(' '.join)

df[['Text_Raw','Text','Text1']]

# 3. Remove Stopwords
SW = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda mail: [w for w in mail if w not in SW])
# 3.1 - Sentences
df['Text1'] = df['Text1'].apply(lambda mail: [w for w in mail.split() if w not in SW]).apply(' '.join)


df[['Text_Raw','Text','Text1']]


# 4. Stemming on the tokens
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
df['Text'] = df['Text'].apply(lambda mail: [stemmer.stem(w) for w in mail])
# 4.1
df['Text1'] = df['Text1'].apply(lambda mail: [stemmer.stem(w) for w in mail.split()])



df[['Text_Raw','Text','Text1']]


# Prepare Data for Modeling 
# ---------------------------------------------------

from sklearn.model_selection import StratifiedShuffleSplit

# Train Test Split
sampler = StratifiedShuffleSplit(n_splits=1,test_size=0.3)
tr_id, va_id = list(*sampler.split(df['Text'].values, df['Label'].values))
tr_df, va_df = df.loc[tr_id,['Text','Label']], df.loc[va_id,['Text','Label']]

# Check class proportions are maintained
print(list(tr_df))
print(tr_df.Label.value_counts().values / len(tr_df))
print(va_df.Label.value_counts().values / len(va_df))