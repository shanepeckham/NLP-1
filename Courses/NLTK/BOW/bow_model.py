
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
BOW Languange Models works with Term-docs frequency matrices.
You can extend what is in the axis (word/sentences or bigram/pages)
'''


# HISTOGRAM OF WORDS
# ------------------

# Dataset
data = nltk.sent_tokenize(paragraph)

# Preprocessing
for i, sentence in enumerate(data):
    # Lowercase
    sentence = sentence.lower()
    # Non-word characters 
    sentence = re.sub(r'\W', ' ', sentence)
    # Spaces
    data[i] = re.sub(r'\s+', ' ', sentence)
    
# Frequencies
word_count = dict()
for sentence in data:
    words = nltk.word_tokenize(sentence)
    for word in words:
        if word not in word_count.keys():
            word_count[word] = 1
        else:
            word_count[word] += 1
            
# Filter top-N words
freq_words = heapq.nlargest(30, word_count, key=word_count.get)

X = list()
for sentence in data:
    vector = list()
    for word in freq_words:
        if word in nltk.word_tokenize(sentence):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)


X = np.asarray(X)


