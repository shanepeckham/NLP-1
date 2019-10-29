
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
from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

workdir = dirname(__file__)
os.chdir(workdir)

# Data Exploratory Analysis
# -------------------------

df = pd.read_csv('spam.csv')    
df.head(2)

# Remove unnecesary columns
df = df.iloc[:,:2]
df.columns = ['Label','Text']

# Convert class to a Label (Factor in R)
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].astype("category")

# Missing Values
df.info()

# Class balanceness
df['Label'].hist()
print(df.Label.value_counts().values / len(df))

# Length of the emails
df['Length'] = df['Text'].apply(lambda x: len(x))
df['Length'].describe()

# Distribution of Lenght of Texts based on their labels
df.groupby('Label').hist(sharex=True, sharey=False, ax=plt.gca())



X, y = df['Text'].to_list(), df['Text'].to_list()
