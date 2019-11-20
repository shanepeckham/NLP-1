
import os
import re
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# DiCaprio Oscars Speech
with open('Courses/NLTK/data/dicaprio.txt') as f:
    paragraph = f.read()


# TOKENIZATION
# ------------

words = nltk.word_tokenize(paragraph)
sentences = nltk.sent_tokenize(paragraph)

## Manually
#cl = lambda x: ' '.join(x.split()) 
#sentences = [cl(sentence.strip()) for sentence in paragraph.split('.')]
#words = [word for sentence in sentences for word in sentence.split(' ')]
#words_in_sentences = [[word for word in sentence.split(' ')] for sentence in sentences]


# STEMMING
# --------

from nltk import stem
stemmer = stem.PorterStemmer()

stems = [stemmer.stem(word) for word in words]
stem_words = [[stemmer.stem(word) for word in sentence.split(' ')] for sentence in sentences]
stem_sentences = [' '.join([stemmer.stem(word) for word in sentence.split(' ')]) for sentence in sentences]


# LEMMATIZATION
# -------------

from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemms = [lemmatizer.lemmatize(word) for word in words]
lemm_words = [[lemmatizer.lemmatize(word) for word in sentence.split(' ')] for sentence in sentences]
lemm_sentences = [' '.join([lemmatizer.lemmatize(word) for word in sentence.split(' ')]) for sentence in sentences]


# STOPWORDS
# ---------
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
sw_sentences = [' '.join([word for word in sentence.split(' ') if word not in STOPWORDS ]) for sentence in sentences]




