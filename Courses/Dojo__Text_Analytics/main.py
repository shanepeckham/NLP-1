
import os
import csv
import nltk

from pathlib import Path
from os.path import abspath, dirname

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

workdir = dirname(__file__)
os.chdir(workdir)

sentences = list()
with open('spam.csv') as f:
    for row in csv.reader(f, delimiter=','):
        sentences.append(row)
    
sentences = [nltk.sent_tokenize(sentence) for sentence in sentences]
