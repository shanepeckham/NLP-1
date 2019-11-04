
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


def order_df_count(df):
    # Order Columns by Frequency of the Word in the Entire Corpus
    return df[df.sum().sort_values(ascending=False).index.to_list()]


def check_differences(s1,s2):
    sortset = lambda l: sorted(sorted(list(l)),key=len, reverse=True)
    s12 = s1 - s2
    s21 = s2 - s1
    s = s1 ^ s2
    print('Elements present in A but not in B: ', sortset(s12))
    print('Elements present in B but not in A: ', sortset(s21))
    print('Elements present in only on of them: ', sortset(s))
    return s


# NLTK Pipeline for text processing
# ---------------------------------

def is_sentence(x):
    return type(x) == str

def is_tokens(x):
    return type(x) == list

def token_to_sentence(x):
    return ' '.join(x)

def sentence_to_token(x,tokenizer=RegexpTokenizer(r'\w+')):
    return tokenizer.tokenize(x)

def ensure_sentence(x):
    return x if is_sentence(x) else token_to_sentence(x)

def preproces_pipeline(
    df:pd.DataFrame, col:str,
    tokenizer, stemmer,
    SW:list, as_tokens:bool=True):
    # Decouple the column to be modified
    ds = df[col]
    ds = ds.apply(ensure_sentence)
    # 1. Lowercase
    ds = ds.apply(lambda mail: mail.lower())
    # 2. Tokenization and Remove Punk
    ds = ds.apply(tokenizer.tokenize)
    # 3. Remove single character words
    ds = ds.apply(lambda mail: [w for w in mail if len(w)>1])
    # 4. Remove Stopwords
    ds = ds.apply(lambda mail: [w for w in mail if w not in SW])
    # 5. Stemming on the tokens
    ds = ds.apply(lambda mail: [stemmer.stem(w) for w in mail])
    # 6. Integrate back and Remove emails where there are no words left
    df.loc[:,col] = ds.values
    df = df[df[col].map(len) > 0].reset_index()
    # 7. Return collection of tokens o collection of sentences    
    return df if as_tokens else df.apply(token_to_sentence)

df = preproces_pipeline(
    df,
    'Text', 
    TOKENIZER,
    STEMMER,
    STOPWORDS,
    as_tokens=True)

df.head()


# Prepare Data for Modeling 
# --------------------------

from sklearn.model_selection import StratifiedShuffleSplit
sampler = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=2019)
tr_id, va_id = list(*sampler.split(df['Text'].values, df['Label'].values))
tr_df, va_df = df.loc[tr_id,['Text','Label']], df.loc[va_id,['Text','Label']]

'''
# BAG OF WORDS MODEL
# ------------------
Create a document-term matrix 
'''

# 1 - SCRATCH
# -----------
''' 
It could make sense to first find the most frequent words:
and then create the BOW for only those. For that, after computing dfm:

freq_words = heapq.nlargest(300, dfm, key=word_frequency.get)
freq_words_counts = sorted(word_frequency.items(), key=lambda k: k[1], reverse=True)

But Dojo just do it for all, so let's do that
'''

from collections import defaultdict
dfm_sc = defaultdict(lambda: defaultdict(int))
for i,email in enumerate(tr_df['Text']):
    for word in email:
        dfm_sc[word][i] += 1 

dfm_sc_df = order_df_count(pd.DataFrame.from_dict(dfm_sc).fillna(0))
dfm_sc_df.head()

# 2 - SKLEARN
# -----------
from sklearn.feature_extraction.text import CountVectorizer
''' 
Convert a collection of text documents to a matrix of token counts 
Implements both tokenization and occurrence counting in a single class
'''
vectorizer = CountVectorizer(
    max_features=None,      # The most common 2000 words
    min_df=1,               # Exclude all that appear in < 3 docs
    max_df=1.,               # Exclude all that appear in > X/100 % docs
    stop_words=stopwords.words('english'))

dfm_sk = vectorizer.fit_transform(tr_df['Text'].apply(token_to_sentence)).toarray()
dfm_sk_df = order_df_count(pd.DataFrame(dfm_sk, columns=vectorizer.get_feature_names()))
dfm_sk_df.head()

# Words that are missing in Sklearn implementation
diff = check_differences(set(dfm_sk_df.columns), set(dfm_sc.keys()))


# 3 - GENSIM
# ----------
from gensim import corpora
dictionary = corpora.Dictionary(tr_df['Text'])

bow = tr_df['Text'].apply(dictionary.doc2bow)
dfm_gs = defaultdict(lambda: defaultdict(int))
for i,l in enumerate(bow):
    for id,count in l:
        dfm_gs[i][dictionary[id]] = count

dfm_gs_df = order_df_count(pd.DataFrame.from_dict(dfm_gs).T.fillna(0))
dfm_gs_df.head()

diff = check_differences(set(dfm_sc.keys()), set(dfm_gs_df.columns))

