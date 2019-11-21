
import os
import pickle
from pprint import pprint
from os.path import join as JP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.nlp_utils import preproces
from utils.general import parse_yaml, ensure_directories

from scripts.catalog import (
    Catalog, Document, Corpus,
    load_catalog, load_corpus)

from utils.datahandling import (
    filter_dict_by_keys, 
    filter_dict_by_vals,
    filter_df_mean_thres)


# CONFIG 
# ------

config = parse_yaml('config.yaml')
paths = config['paths']
ensure_directories(paths)

DATA_PATH = paths['data']
TOPIC = config['topic']
CLASSES = config['classes']

with open(paths['stopwords'], 'rb') as f:
    SW = pickle.load(f)


# IMPORT / SAVE CATALOG TO USE
# ----------------------------

catalog = Catalog()
corpus = load_corpus(path=paths['catalog'], name='corpus1')
catalog.load_corpus(corpus=corpus)
# catalog.save(path=paths['catalog'],name='test1_clean')

OF_INTEREST = ['AU', 'BR', 'CA', 'CN', 'WO', 'US']

# Filter down the catalog
filters = dict(
    topic = ['isocyanate'],
    country = ['CN'],       # country = OF_INTEREST
    raw_text_len = 100)

catalog_sample = catalog.filter_catalog(filters)
print('Catalog recuded from {} to {}'.format(
    len(catalog.documents), len(catalog_sample.documents)))



# SPLIT CATALOG INTO THE TWO CATEGORIES
# -------------------------------------

filters = dict(label='relevant')
pos_catalog = catalog_sample.filter_catalog(filters)

filters = dict(label='irrelevant')
neg_catalog = catalog_sample.filter_catalog(filters)

print('Positive documents: ',len(pos_catalog.documents))
print('Negative documents: ',len(neg_catalog.documents))




# MODELING 
# ------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

pos_catalog.collect_corpus()
neg_catalog.collect_corpus()
corpus = catalog_sample.collect_corpus(type_='list')

'''
BAG OF WORDS
------------
'''

vectorizer = CountVectorizer(
    min_df=3,
    max_df=.7,
    lowercase=True,
    ngram_range=(1,3),
    max_features=3000,
    stop_words=SW)

pos_bow = pos_catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='BOW',
    max_docs=10)

neg_bow = neg_catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='BOW',
    max_docs=10)

pos_bow.representation.head()
pos_catalog.models


'''
TF-IDF
------
'''

vectorizer = TfidfVectorizer(
    min_df=.1,
    max_df=.7,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    max_features=3000,
    ngram_range=(1,3),
    lowercase=True,
    stop_words=SW)

tfidf = catalog_sample.to_matrix(
    vectorizer=vectorizer,
    modelname='TFIDF',
    max_docs=50)

pos_tfidf = pos_catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='TFIDF',
    max_docs=50)

neg_tfidf = neg_catalog.to_matrix(
    vectorizer=vectorizer,
    modelname='TFIDF',
    max_docs=50)

pos_tfidf.representation.head()
pos_tfidf_df = pd.DataFrame(
    {"word": [k for k,v in pos_tfidf.token2id.items()],
     "idf": [pos_tfidf.mapping.idf_[v] \
         for k,v in pos_tfidf.token2id.items()]}) \
        .sort_values("idf",ascending=False)
pos_tfidf_df.head()


neg_tfidf.representation.head()
neg_tfidf_df = pd.DataFrame(
    {"word": [k for k,v in neg_tfidf.token2id.items()],
     "idf": [neg_tfidf.mapping.idf_[v] \
         for k,v in neg_tfidf.token2id.items()]}) \
        .sort_values("idf",ascending=False)
neg_tfidf_df.head()





# #### CLEAN THIS BEFORE MERGE ####

from scripts.algorithms.clustering import (
    ward_clustering, 
    plot_dendogram_from_linkage_matrix
)


plot_dendogram_from_catalog(
    pos_catalog.models['TFIDF'], n_terms=20)





# pos_catalog.models['TFIDF'].representation.shape
# pos_catalog.models['TFIDF'].representation.head()

# mat = ward_clustering(
#     pos_catalog.models['TFIDF'], n_terms=20)

# labels = get_most_relevant_terms(
#     pos_catalog.models['TFIDF'], n_terms=20)

# plot_dendogram_from_linkage_matrix(
#     linkage_matrix=mat,
#     clusters=10,
#     truncate_mode=None,
#     labels=labels,
#     orientation='right'
# )


# _, ax = plt.subplots(figsize=(15, 20)) # set size
# dendrogram(
#     Z=mat,
#     p=10,
#     truncate_mode=None,  # show only the last p merged clusters
#     orientation='right', 
#     labels=labels, 
#     ax=ax)

# plt.tick_params(
#     axis= 'x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off')

# plt.tight_layout() #show plot with tight layout
# plt.show()