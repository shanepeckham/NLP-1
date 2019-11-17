

'''
Process Text from multiple sources and formats.
Convert data between plain Python, Pandas for Scikit-Learn and Gensim Data Structures

'''

import numpy as np
from pprint import pprint
from nltk.corpus import stopwords
from collections import defaultdict
from language.utils import token_to_sentence


import gensim
import pandas as pd
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


''' 
                            DEFINITIONS
                            ===========

Kind of Inputs when dealing with Text
--------------------------------------
--------------------------------------

Input is a plaint text   --> We read it generating a list of list --> txtdoc
......................
The first list is the document-list. The nested lists are the sentences-list. 
    
    [
        ['This is the sentence 1 of document 1',     Sentence 1
         '------------------------------------',     Sentence n       Doc 1
         'This is the sentence N of document 1       Sentence N
        ],
     
        ['------------------------------------'],                     Doc d
     
        ['This is the sentence 1 of document D'],
         '------------------------------------',                      Doc D
         'This is the sentence N of document D'
        ]
    ]


Input is a CSV file   --> We read it using pandas  --> pdDoc
...................
There is a column that could contain the label (text classification)
There is a column that containes each of the documents. Pandas Series



CORPUS Representations / Data Structures
----------------------------------------
----------------------------------------

Plain Python: --> Dictionary

    - Reads the Nested Lists Input
    - Look at BoW Resprensentation since Python doesn't have a Data Structure for this


Pandas: --> Pandas Series

    - Reads the Pandas Dataframe Input
    - The column which context the text will be the Pandas Series representing the Corpus
    - Each row contains each of the documents

Gensin: --> corpora.dictionary.Dictionary

    - Reads the Nested Lists Input
    - Create a list of tuples, where:
        - Each index of the list represents the index of each of the documents and stores a tuple
        - Each tuple consist of (token_id, counts of token_id in document of that index)
            - Dictionary has a method to map token ids to word token2id() and id2token()




Bag of Words Model
------------------

Plain Python: --> dictBow

    What is its representation?
        Dictionary where each key is a unique token, and a value is a new dictionary,
        where the new key is the document id and the value is the # of times it appears in it

    How to build it?

        If input == list of list.
            Iterate over the words of the entire corpus and calculate the frequency of each words
            taking into account the words and the document in which it appeared.

        If input == CSV file
            Use Pandas representation for importing and then transform it using ` def() `

Pandas - Sklearn: --> pdBow

    What is the representation?
        Pandas Dataframe representing the Document-Term-Frequency-Matrix.
        Rows represent the docuemnt id and Columns represent each of the tokens.



Gensim:  --> genBow

    Gensim has a gensim.corpora.dictionary.Dictionary class object which stores ids<->tokens
    The function doc2bow() to t 




TF-IDF Model
------------

Plain Python:




'''




''' 
            DATA COMES AS A CSV 
            ===================    
'''
    

# Plain Python
def pdDoc2dictBow(doc:pd.Series):
    ''' Transform a document as a pd.Series 
    into a python dictionary Bag Of Words Model '''
    bow = defaultdict(lambda: defaultdict(int))
    for d,sentence in enumerate(doc):
        for word in sentence:
            bow[word][d] += 1
    return bow

# Pandas
def pdDoc2pdBow(
    doc:pd.Series,
    index:pd.Series=None,
    fillna=int(0)):
    ''' Transform a Corpus as pd.Series 
    into a pd.Dataframe Bag Of Words Model '''
    dic = pdDoc2dictBow(doc)
    if index is not None:
        return pd.DataFrame.from_dict(dic).fillna(0)
    return pd.DataFrame.from_dict(dic).fillna(0).set_index(index)

def dictBow2pdBow(
    dict:dict):
    return

def csv2Bow(document_path:str, col:str):
    ''' This function is going to be too customized 
    Only for cases where you know the csv and the column '''
    df = pd.read_csv(document_path)
    return pdDoc2pdBow(df[col])

def term_frequency(row):
    ''' Normalize the frequency of the row 
    with the total number of records with a count >= 1'''
    return row/np.sum(row)

def inverse_doc_frequency(column):
    ''' Weight each value of a column with the lenght of the column /
    the sum of all records with a count >= 1 '''
    return np.log10(len(column)/np.sum(column>0))

def pdBow2dictTfidf(df:pd.DataFrame):
    tf = df.apply(term_frequency, axis=1)
    idf = df.apply(inverse_doc_frequency, axis=0)
    return idf * tf




''' 
        Sklearn 
        =======
'''

''' BOW '''

default_vectorizer = CountVectorizer(
    max_features=None,
    min_df=1,              
    max_df=1.,
    stop_words=stopwords.words('english'))

def pdDoc2pdBow_sklearn(
    df:pd.Series,
    vectorizer:CountVectorizer=default_vectorizer):
    return pd.DataFrame(
        vectorizer.fit_transform(df.apply(token_to_sentence)).toarray(),
        columns=vectorizer.get_feature_names()
    )

''' TFIDF '''

default_tfidf_vectorizer = TfidfVectorizer(
        max_features=None,
        min_df=1,
        max_df=1.,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        ngram_range=(1,1),
        stop_words=stopwords.words('english'))

def pdDoc2pdTfidf_sklearn(
    df:pd.Series,
    vectorizer:TfidfVectorizer=default_tfidf_vectorizer):
    return pd.DataFrame(
        vectorizer.fit_transform(df.apply(token_to_sentence)).toarray(),
        columns=vectorizer.get_feature_names().reindex((tfidfm_sc.columns), axis=1)
    )


# def pdDoc2sklearnBow(doc:pd.Series,):
#     pass
    

''' 
        Gensim 
        ======
'''

def pdDocgenDict2dictBow(
    pdDoc:pd.Series, 
    dictionary:gensim.corpora.dictionary.Dictionary):
    ''' Tranform pandas Doc Representation to a Dict '''
    bow = pdDoc.apply(dictionary.doc2bow)
    genbow = defaultdict(lambda: defaultdict(int))
    for doc_id, doc_values in enumerate(bow):
        for token_id, token_count in enumerate(doc_values):
            genbow[doc_id][dictionary[token_id]] = token_count
    return genbow

def genDoc2pdBow(
    pdDoc:pd.Series, 
    dictionary:gensim.corpora.dictionary.Dictionary,
    index=None,
    fillna=0):
    ''' Tranforms a Pandas Doc with a Gensim doc to a Pandas BOW''' 
    dict_bow = pdDocgenDict2dictBow(pdDoc, dictionary)
    if index is not None:
        return pd.DataFrame.from_dict(dict_bow).fillna(fillna)
    return pd.DataFrame.from_dict(dict_bow).fillna(fillna).set_index(index)

# dfm_gs = gensinBow2dict(dictionary)
# dfm_gs_df = gensinBow2pandas(bow=dictionary,index=None,fillna=0.)







