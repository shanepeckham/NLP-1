
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import spacy
# nlp = spacy.load("data/lang_models/english_sm")
# nlp = spacy.load("en_core_web_sm")

def strip(l):
    return l.strip()

def preproces(l):
    l = strip(l)
    return l

def is_sentence(s):
    s = preproces(s)
    # Case sentence of 1 word or less
    if len(s.split(' ')) <= 1:
        return False
    # Case only digits:
    if s.isdigit():
        return False
    return True

def token_to_sentence(x):
    return ' '.join(x)

def sentence_to_token(x,tokenizer=RegexpTokenizer(r'\w+')):
    return tokenizer.tokenize(x)

def ensure_sentence(x):
    return x if is_sentence(x) else token_to_sentence(x)

def process_text(text, lemmatizer=WordNetLemmatizer(), stemmer=PorterStemmer()):
    ''' 
    Simple preliminar text processing pipeline
    1 - Lowercase
    2 - Tokenization
    3 - Lemmatization
    4 - Stemming
    5 - Remove words of only 1 letter
    '''    
    text = text.lower()
    text = sentence_to_token(text)
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [stemmer.stem(word) for word in text]
    text = [word for word in text if len(word)>1]
    return text

