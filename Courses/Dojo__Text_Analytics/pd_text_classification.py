
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

from language.utils import check_differences, order_df_count, preproces_pipeline

os.chdir(dirname(__file__))

# CONFIG
# ------
TOKENIZER = RegexpTokenizer(r'\w+')
STEMMER = PorterStemmer()
STOPWORDS = stopwords.words('english')



'''

# DATA INGESTION
# -------------------------------------------------------------------


'''

df = pd.read_csv('spam.csv')    
df = df.iloc[:,:2]
df.columns = ['Label','Text']

df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].astype("category")
categories = pd.Categorical(df['Label']).categories


# NLTK Pipeline for text processing
# ---------------------------------

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
# -------------------------------------------------------------------
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

from language.py_pd_gen import pdDoc2dictBow, pdDoc2pdBow
dfm_sc = pdDoc2dictBow(doc=tr_df['Text'])
dfm_scracth_df = pdDoc2pdBow(doc=tr_df['Text'],index=tr_df['index'],fillna=0)



# 2 - SKLEARN
# -----------
''' 
Convert a collection of text documents to a matrix of token counts 
Implements both tokenization and occurrence counting in a single class
'''
from sklearn.feature_extraction.text import CountVectorizer
from language.py_pd_gen import pdDoc2pdBow_sklearn

dfm_sklearn_df = pdDoc2pdBow_sklearn(
    df=tr_df['Text'],
    vectorizer=CountVectorizer(
        max_features=None,      # The most common 2000 words
        min_df=1,               # Exclude all that appear in < 3 docs
        max_df=1.,               # Exclude all that appear in > X/100 % docs
        stop_words=stopwords.words('english')))

# Words that are missing in Sklearn implementation
# We remove the columns to make fair comparisons later on 
diff = check_differences(set(dfm_sc.keys()),set(dfm_sklearn_df.columns))
dfm_scracth_df.drop(list(diff),axis=1,inplace=True)



# # 3 - GENSIM
# # ----------
import gensim
from gensim import corpora
from collections import defaultdict
# from language.py_pd_gen import pdDocgenDict2dictBow, genDoc2pdBow
# from language.py_pd_gen import genDict2dictBow, gensinBow2pandas

dictionary = corpora.Dictionary(tr_df['Text'])
bow = tr_df['Text'].apply(dictionary.doc2bow)
dfm_gs = defaultdict(lambda: defaultdict(int))
for i,l in enumerate(bow):
    for id,count in l:
        dfm_gs[i][dictionary[id]] = count

dfm_gs_df = pd.DataFrame.from_dict(dfm_gs).T.fillna(0)
order_df_count(dfm_gs_df).head()
dfm_gs_df.head()

diff = check_differences(set(dfm_sc.keys()), set(dfm_gs_df.columns))



# Plain Gensim
# dense_vector= matutils.corpus2dense(sparse_vector,num_terms=len(dictionary.token2id))





'''


TF-IDF MODEL
# -------------------------------------------------------------------

Extends BOW model taking into account frequency of words 
as a measure of the importance of the terms.



'''

# 1 - SCRATCH
# -----------

from language.py_pd_gen import pdBow2dictTfidf
tdidf_scratch_df = pdBow2dictTfidf(dfm_scracth_df)
# Check --> tfm.iloc[0,:].replace(0,np.nan).dropna()
# order_df_count(tfidfm).head()
# order_df_count(tfidfm.T).head()


# 2 - SKLEARN
# -----------

from language.py_pd_gen import pdDoc2pdBow_sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

dfm_sklearn_df = pdDoc2pdBow_sklearn(
    df=tr_df['Text'],
    vectorizer=tf_transformer(
        max_features=None,
        min_df=1,
        max_df=1.,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        ngram_range=(1,1),
        stop_words=stopwords.words('english')))

dfm_sklearn_df = dfm_sklearn_df.reindex((tdidf_scratch_df.columns), axis=1)

dfm_sklearn_df.T.head()
# order_df_count(tfm_sklearn_df).head()


# 3 - GENSIM
# ----------
from gensim.models import TfidfModel
tf_gensim = TfidfModel(bow.tolist(), smartirs='ntc')
tfm_g = tf_gensim[bow.tolist()]

for document in tfm_g:
    for id,freq in document:
        print([dictionary[id], np.around(freq, decimals=2)])
