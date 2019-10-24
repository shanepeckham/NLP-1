
'''
LATENT SEMANTIC ANALYSIS
========================

Analysis of relationships between a set of documents
and the terms they content by producing a set of concepts
related to the documents and terms
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample Data
dataset = ["The amount of polution is increasing day by day",
           "The concert was just great",
           "I love to see Gordon Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are examples of great technology present today",
           "All of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"]

dataset = [line.lower() for line in dataset]


# TFIDF creates the TF-IDF Model given more functionalities
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# SVD Decomposition --> n_componentes == topics
lsa = TruncatedSVD(n_components=4, n_iter=100)
lsa.fit(X)

row1 = lsa.components_[0]
