import os
import pickle
from pprint import pprint
from os.path import join as JP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import string

from utils.nlp_utils import preproces
from utils.general import parse_yaml
from scripts.catalog import Catalog, load_catalog, load_corpus

import spacy
from spacy import displacy



config = parse_yaml('config.yaml')
paths = config['paths']
catalog = Catalog()
catalog = load_catalog(path=paths['catalog'], name='spacy_pipeline_on_US_corpus')
print(len(catalog.documents))


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.algorithms.clustering import (
    kmean_clustering, 
    plot_clusters_as_wordclouds)


''' TFIDF '''
vectorizer = TfidfVectorizer(
    min_df=.1,
    max_df=.7,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    max_features=3000,
    ngram_range=(1,3),
    lowercase=True,
    stop_words=stopwords.words('english'))

catalog.collect_corpus(attr='processed_text', form=list)
tfidf = catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='TFIDF',
    max_docs=50)

tfidf.representation.head()

'''
FLAT CLUSTERING
---------------
'''
NUM_CLUSTERS = 4
EMBED_SIZE = 10000
WORDS_PER_CLUSTER = 50

clustered_words = kmean_clustering(
    model=catalog.models['TFIDF'],
    num_clusters=NUM_CLUSTERS, 
    words_per_cluster=WORDS_PER_CLUSTER)

''' Clustering2WordCloud '''
plot_clusters_as_wordclouds(tfidf, clustered_words, method='idf')