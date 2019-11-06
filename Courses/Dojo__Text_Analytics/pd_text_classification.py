
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

from utils import check_differences, order_df_count
from py_pd_gen import preproces_pipeline

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
--
freq_words = heapq.nlargest(300, dfm, key=word_frequency.get)
freq_words_counts = sorted(word_frequency.items(), key=lambda k: k[1], reverse=True)
--
But Dojo just do it for all, so let's do that
'''

from collections import defaultdict

def doc2dictBOW(doc:pd.Series):
    ''' Transform a document into a python dictionary Bag Of Words Model '''
    bow = defaultdict(lambda: defaultdict(int))
    for d,sentence in enumerate(doc):
        for word in sentence:
            bow[word][d] += 1
    return bow

def doc2dfBOW(
    doc:pd.Series,
    index:pd.Series=None,
    fillna=int(0)):
    ''' Transform a document into a Pandas Dataframe Bag Of Words Model '''
    dic = doc2dictBOW(doc)
    if index is not None:
        return pd.DataFrame.from_dict(dic).fillna(0)
    return pd.DataFrame.from_dict(dic).fillna(0).set_index(index)

dfm_sc = doc2dictBOW(doc=tr_df['Text'])
dfm_sc_df = doc2dfBOW(doc=tr_df['Text'],index=tr_df['index'],fillna=0)
pprint(dfm_sc)
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

def gensinBow2dict(
    bow:pd.Series, 
    dictionary:gensim.corpora.dictionary.Dictionary):
    ''' Tranform Gensim BOW Representation to a Dict '''
    genbow = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_values in enumerate(bow):
        for token_id, token_count in doc_values:
            genbow[doc_id][dictionary[id]] = token_count
    return genbow

def gensinBow2pandas(doc,index=None,fillna=0):
    dict_bow = gensinBow2dict(doc)
    if index is not None:
        return pd.DataFrame.from_dict(dict_bow).fillna(fillna)
    return pd.DataFrame.from_dict(dict_bow).fillna(fillna).set_index(index)

dfm_gs = gensinBow2dict(dictionary)
dfm_gs_df = gensinBow2pandas(bow=dictionary,index=None,fillna=0.)
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
def term_frequency(row):
    ''' Normalize the frequency of the row 
    with the total number of records with a count >= 1'''
    return row/np.sum(row)

def inverse_doc_frequency(column):
    ''' Weight each value of a column with the lenght of the column /
    the sum of all records with a count >= 1 '''
    return np.log10(len(column)/np.sum(column>0))


tfm_sc = dfm_sc_df.apply(term_frequency, axis=1)
idfm_sc = dfm_sc_df.apply(inverse_doc_frequency, axis=0)
tfidfm_sc = idfm_sc * tfm_sc
# Check --> tfm.iloc[0,:].replace(0,np.nan).dropna()

# tfm_sc.head()
# idfm_sc.head()
tfidfm_sc.head()
tfidfm_sc.T.head()
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
tfm_sk_df = tfm_sk_df.reindex((tfidfm_sc.columns), axis=1)

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
