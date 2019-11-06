
'''
Process Text from multiple sources and formats.
Convert data between plain Python, Pandas for Scikit-Learn and Gensim Data Structures

'''

import os
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from os.path import dirname
os.chdir(dirname(__file__))


''' DATA COMES AS A CSV '''

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