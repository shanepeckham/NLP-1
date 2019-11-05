
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


def order_df_count(df, col=None):
    # Order Columns by Frequency of the Word in the Entire Corpus
    if not col:
        return df[df.sum().sort_values(ascending=False).index.to_list()]
    return df[df[col].sort_values(ascending=False).index.to_list()]


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
    df = df[df[col].map(len) > 0].reset_index(drop=True)
    # 7. Return collection of tokens o collection of sentences    
    return df if as_tokens else df.apply(token_to_sentence)

df = preproces_pipeline(
    df,
    'Text', 
    TOKENIZER,
    STEMMER,
    STOPWORDS,
    as_tokens=True)

df['index'] = ['mail_{}'.format(i) for i in range(len(df))]
df.head()

# Prepare Data for Modeling 
# --------------------------

from sklearn.model_selection import StratifiedShuffleSplit
sampler = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=2019)
tr_id, va_id = list(*sampler.split(df['Text'], df['Label']))
tr_df, va_df = df.loc[tr_id,['index','Text','Label']], df.loc[va_id,['index','Text','Label']]
tr_df.head()

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

def doc2bow(doc):
    ''' Transform a document into a BOW representation (dict) '''
    bow = defaultdict(lambda: defaultdict(int))
    for d,sentence in enumerate(doc):
        for word in sentence:
            bow[word][d] += 1
    return bow

def dict2df(dic,index=None,fillna=0):
    if index is not None:
        return pd.DataFrame.from_dict(dic).fillna(0)
    return pd.DataFrame.from_dict(dic).fillna(0).set_index(index)

dfm_sc = doc2bow(tr_df['Text'])
dfm_sc_df = dict2df(dfm_sc,index=tr_df['index'],fillna=0)

dfm_sc_df.head()
order_df_count(dfm_sc_df).head()


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
dfm_sk_df = pd.DataFrame(dfm_sk, columns=vectorizer.get_feature_names())
order_df_count(dfm_sk_df).head()
dfm_sk_df.head()

# Words that are missing in Sklearn implementation
diff = check_differences(set(dfm_sc.keys()),set(dfm_sk_df.columns))
''' We remove the columns to make fair comparisons later on '''
dfm_sc_df.drop(list(diff),axis=1,inplace=True)
len(np.unique(dfm_sk_df.columns))
len(np.unique(dfm_sc_df.columns))


# 3 - GENSIM
# ----------
import gensim
from gensim import corpora

dictionary = corpora.Dictionary(tr_df['Text'])
bow = tr_df['Text'].apply(dictionary.doc2bow)
dfm_gs = defaultdict(lambda: defaultdict(int))
for i,l in enumerate(bow):
    for id,count in l:
        dfm_gs[i][dictionary[id]] = count

def gensinBOW2dict(
    bow:pd.Series, 
    dictionary:gensim.corpora.dictionary.Dictionary):
    ''' Tranform Gensim BOW Representation to a Dict '''
    genbow = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_values in enumerate(bow):
        for token_id, token_count in doc_values:
            genbow[doc_id][dictionary[id]] = token_count
    return genbow

def gensinBOW2pandas(dic,index=None,fillna=0):
    if index is not None:
        return pd.DataFrame.from_dict(dic).fillna(0)
    return pd.DataFrame.from_dict(dic).fillna(0).set_index(index)

dfm_gs_df = pd.DataFrame.from_dict(dfm_gs).T.fillna(0)
order_df_count(dfm_gs_df).head()
dfm_gs_df.head()

diff_ = check_differences(set(dfm_sc.keys()), set(dfm_gs_df.columns))



'''
TF-IDF MODEL
# ----------

Extends BOW model taking into account frequency of words 
as a measure of the importance of the terms.

'''

# 1 - SCRATCH
# -----------
# TF -> Calculate Relative Term Frequency
tf_f = lambda row: row/np.sum(row)  
idf_f = lambda col: np.log10(len(col)/np.sum(col>0))
# idf_f = lambda col: np.log(len(col)/np.sum(col))  # --> Wrong, it is not the sum, but the rows different to zero right?

tfm = dfm_sc_df.apply(tf_f, axis=1)
idfm = dfm_sc_df.apply(idf_f, axis=0)
# Check --> tfm.iloc[0,:].replace(0,np.nan).dropna()

tfm.head()
order_df_count(tfm).head()
tfm.T.head()
idfm.head()

tfidfm = idfm * tfm
tfidfm.T.head()
# order_df_count(tfidfm).head()
# order_df_count(tfidfm.T).head()


# 2 - SKLEARN
# -----------
# TfidfVectorizer == CountVectorized + TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
tf_transformer = TfidfVectorizer(
    max_features=None,
    min_df=1,
    max_df=1.,
    norm=None,
    use_idf=True,
    smooth_idf=True,
    ngram_range=(1,1),
    stop_words=stopwords.words('english'))

tfm_sk = tf_transformer.fit_transform(tr_df['Text'].apply(token_to_sentence)).toarray()
tfm_sk_df = pd.DataFrame(tfm_sk, index=tr_df['index'], columns=tf_transformer.get_feature_names())
tfm_sk_df = tfm_sk_df.reindex((tfidfm.columns), axis=1)

tfm_sk_df.T.head()
# order_df_count(tfm_sk_df).head()


# 3 - GENSIM
# ----------
from gensim.models import TfidfModel
tf_gensim = TfidfModel(bow.tolist(), smartirs='ntc')
tfm_g = tf_gensim[bow.tolist()]

for document in tfm_g:
    for id,freq in document:
        print([dictionary[id], np.around(freq, decimals=2)])
