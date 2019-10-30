
import os
import re
import nltk
import heapq
import numpy as np
from pprint import pprint

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

os.chdir('/Users/pabloruizruiz/OneDrive/Courses/NLP Stanford Course/')
print(os.getcwd())

with open('Courses/NLTK/data/dicaprio.txt') as f:
    dataset = f.read()


'''
BOW Languange Models works with Term-docs frequency matrices.
You can extend what is in the axis (word/sentences or bigram/pages)
'''


# HISTOGRAM OF WORDS
# ------------------

sentences = nltk.sent_tokenize(dataset)

# Preprocessing
for i, sentence in enumerate(sentences):
    # Lowercase
    sentence = sentence.lower()
    # Non-word characters 
    sentence = re.sub(r'\W', ' ', sentence)
    # Spaces
    sentences[i] = re.sub(r'\s+', ' ', sentence)
    
# Frequencies
word_count = dict()
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] += 1
            
# Filter top-N words
freq_words = heapq.nlargest(30, word_count, key=word_count.get)

X = list()
for sentence in sentences:
    vector = list()
    for word in freq_words:
        if word in nltk.word_tokenize(sentence):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)

X = np.asarray(X)


'''
BOW PROBLEMS
============

It assumes the same importance of all words are the same.
But in "She is beautiful", beautiful is way more significant.
We need a way to weight better uncommon (important) words.

TF -IDF 
=======

TF: Term Frequency (doc, word)
IDF: Inverse of Document Frequency (word)
TF-IDF = TF*IDF
'''

# IDF

word_idfs = dict()

for word in freq_words:
    doc_count = 0
    for doc in sentences:
        if word in nltk.word_tokenize(doc):
            doc_count += 1
    word_idfs[word] = np.log((len(sentences)/doc_count)+1)

pprint(word_idfs)

# TF

word_tfs = dict()

for word in freq_words:
    doc_tf = list()
    for sentence in sentences:
        term_count = 0  
        for w in nltk.word_tokenize(sentence):
            if w == word:
                term_count += 1
        tf = term_count/len(nltk.word_tokenize(sentence))
        doc_tf.append(tf)
    word_tfs[word] = doc_tf

pprint(word_tfs)

# TF·IDF

tfidf_matrix = list()
for word in word_tfs:
    tfidf = []
    for value in word_tfs[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)
        
X = np.asarray(tfidf_matrix).T
