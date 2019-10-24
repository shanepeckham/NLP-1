
'''
LATENT SEMANTIC ANALYSIS
========================

Analysis of relationships between a set of documents
and the terms they content by producing a set of concepts
related to the documents and terms
'''

import nltk
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
# This has the probability of each of the (42) words belonging to topic 1
# The name of each of this terms we can retrieve:
terms = vectorizer.get_feature_names()

concept_words = dict()
# For each of the 4 singular components
for i,comp in enumerate(lsa.components_):
    # Get the 42 tuples (term, value of the term)
    componentTems = zip(terms,comp)
    sortedTerms = sorted(componentTems, key=lambda x:x[1], reverse=True)
    # Keep only the 10 most impotant
    sortedTerms = sortedTerms[:10]
    # STORE
    concept_words[('Concept '+str(i))] = sortedTerms


# Training

for concept in concept_words:
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_score in concept_words[concept]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
        sentence_scores.append(score)
    print('\n'+concept+':')
    for sentence_score in sentence_scores:
        print(sentence_score)

