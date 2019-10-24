
import re
import pickle
import numpy as np

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.datasets import load_files

LOAD=True

if not LOAD:
    # Importing dataset
    reviews = load_files('data/review_polarity/txt_sentoken')
    X,y = reviews.data, reviews.target

    with open('data/review_polarity/txt_sentoken/X.pickle', 'wb') as f:
        pickle.dump(X,f)

    with open('data/review_polarity/txt_sentoken/y.pickle', 'wb') as f:
        pickle.dump(y,f)

else:
    # Unpickle dataset
    with open('data/review_polarity/txt_sentoken/X.pickle', 'wb') as f:
        X = pickle.load(f)

    with open('data/review_polarity/txt_sentoken/y.pickle', 'wb') as f:
        y = pickle.load(f)

# Creating the corpus - Preprocessing

corpus = list()
for i in range(len(X)):
    # Every non-word character
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    # Remove single-character words (I, a, ...)
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    # Remove single-character as the start of the sentence
    review = re.sub(r'^[a-z]\s+', ' ', review)
    # Remove the extra spaces generated
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

# Create the Language Model - BOW

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
    max_features=2000,      # The most common 2000 words
    min_df=3, max_df=0.6,    # Excludo all thosethat appear in < 3 and > 60% docs
    stop_words=stopwords.words('english'))

X = vectorizer.fit_transform(corpus).toarray()

# Convert BOW to TF-IDF

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# Training and Testing Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,train_size=0.8)

