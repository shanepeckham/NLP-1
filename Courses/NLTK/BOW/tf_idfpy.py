
import re
import nltk
import heapq
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

with open('../text.txt') as f:
    paragraph = f.read()


'''
PROBLEMS OF BOW MODEL
=====================

It is assigning the same importance to every word


'''


# IDF Dictionary
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = np.log(len(dataset)/(1+doc_count))