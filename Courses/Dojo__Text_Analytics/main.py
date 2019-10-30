
import os
import re
import csv
import heapq
import numpy as np
import pandas as pd
from pprint import pprint
from os.path import abspath, dirname

import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

workdir = dirname(__file__)
os.chdir(workdir)

# Data
# ----

mails = list()
with open('spam.csv') as f:
    for row in csv.reader(f, delimiter=','):
        mails.append(row)
    
df = pd.DataFrame(mails)
df = df.drop([2,3,4],axis=1).drop(0)
df.columns = ['Label','Text']
df['Label'].hist()
X, y = df['Text'].to_list(), df['Text'].to_list()


# Preprocessing
# -------------

STOPWORDS = stopwords.words('english')

mails = list()
for i in range(len(X)):
    mail = X[i]
    # Lowercase
    mail = mail.lower()
    # Non-word characters 
    mail = re.sub(r'\W', ' ', mail)
    # Digits
    mail = re.sub(r'\d', ' ', mail)
    # Remove single-character words (I, a, ...)
    mail = re.sub(r'\s+[a-z]\s+', ' ', mail)
    # Remove single-character as the start of the mail
    mail = re.sub(r'^[a-z]\s+', ' ', mail)
    # Remove the extra spaces generated
    mail = re.sub(r'\s+', ' ', mail)
    # Rmove spaces
    mail = re.sub(r'\s+', ' ', mail)
    # Lemmatize
    mails.append(lemmatizer.lemmatize(mail))


# BAG OF WORDS - Of only the N most frequent words
# ------------

# 1 - Discover these frequent words
word_count = dict()
for mail in mails:
    for word in nltk.word_tokenize(mail):
        if word not in STOPWORDS:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

freq_words = heapq.nlargest(30, word_count, key=word_count.get)
pprint(freq_words)

# 2 - Create the doc-term matrix
dt_matrix = list()
for mail in mails:
    vector = list()
    for word in freq_words:
        if word in nltk.word_tokenize(mail):
            vector.append(1)
        else:
            vector.append(0)
    dt_matrix.append(vector)
df_matrix = np.asarray(dt_matrix)
