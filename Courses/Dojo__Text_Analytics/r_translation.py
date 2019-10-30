
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
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

os.chdir(dirname(__file__))


# CONFIG
# ------
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')


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

def preproces_pipeline(df:pd.Series, tokenizer, stemmer, SW:list):
    # 1. Lowercase
    df = df.apply(lambda x: x.lower())
    # 2. Tokenization and Remove Punk
    df = df.apply(tokenizer.tokenize)
    # 3. Remove Stopwords
    df = df.apply(lambda mail: [w for w in mail if w not in SW])
    # 4. Stemming on the tokens
    df = df.apply(lambda mail: [stemmer.stem(w) for w in mail])
    return df

df['Text'] = preproces_pipeline(
    df['Text'], 
    TOKENIZER,
    STEMMER,
    STOPWORDS)


# Prepare Data for Modeling 
# ---------------------------------------------------

from sklearn.model_selection import StratifiedShuffleSplit
sampler = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=2019)
tr_id, va_id = list(*sampler.split(df['Text'].values, df['Label'].values))
tr_df, va_df = df.loc[tr_id,['Text','Label']], df.loc[va_id,['Text','Label']]


# BAG OF WORDS MODEL
# ------------------
'''
Create a document-term matrix
'''

# 1 - SCRATCH 
''' 
It could make sense to first find the most frequent words:
and then create the BOW for only those. For that:

from collections import defaultdict
word_frequency = defaultdict(int)
for sample in tr_df['Text'].values:
    for word in sample:
        word_frequency[word] += 1

# The 30 most frequent words
freq_words = heapq.nlargest(300, word_frequency, key=word_frequency.get)
freq_words_counts = sorted(word_frequency.items(), key=lambda k: k[1], reverse=True)

But Dojo just do it for all, so let's do that
'''


dfm = dict()
tr_emails = tr_df['Text'].to_list()
for i,email in enumerate(tr_emails):
    for word in email:
        if word not in dfm.keys():
            dfm[word] = dict()
        if i not in dfm[word].keys():
            dfm[word][i] = 1
        else:
            dfm[word][i] += 1

dfm_ = pd.DataFrame.from_dict(dfm).fillna(0)
dfm_.head()


# 2 - SCIKIT-LEARN
from sklearn.feature_extraction.text import CountVectorizer
''' 
Convert a collection of text documents to a matrix of token counts 
Implements both tokenization and occurrence counting in a single class
'''
vectorizer = CountVectorizer(
    max_features=2000,      # The most common 2000 words
    min_df=3,               # Exclude all that appear in < 3 docs
    max_df=0.6,             # Exclude all that appear in > 60% docs
    stop_words=stopwords.words('english'))


X = vectorizer.fit_transform(
    tr_df['Text'].apply(lambda x: ' '.join(x))).toarray()








# 3 - GENSIM
from gensim import corpora
dictionary = corpora.Dictionary(df['Text'])

