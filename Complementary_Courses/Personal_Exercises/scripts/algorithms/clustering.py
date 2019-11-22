

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity

import requests
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

'''
REGULAR CLUSTERING 
==================
'''

''' ALGORITHM '''

def kmean_clustering(
    model, #:Model
    num_clusters:int, 
    words_per_cluster:int=None):
    '''
    Clusters using k-mean with k words per cluster
    ----------------------------------------------
    The k-words are found as followed:
        - 1: Filter by minimum idf -> Maximum importance
        - 2: Return the k closest to the centroid of that cluster

    Inputs:
    -------
        - model: Trained instance of class Model
        - num_clusters: Number of Clusters to look for
        - words_per_cluster: K parameter above

    Returns:
    -------- 
        - Dict key='cluster id', value=k_words_closest_to_centroid
    '''
    km = KMeans(n_clusters=num_clusters)
    km.fit_transform(model.representation)
    # clusters = km.labels_.tolist()

    # Filter K most similar words to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

    cluster_words = defaultdict(list)
    for i in range(num_clusters):
        for idx in order_centroids[i, :words_per_cluster]:
            cluster_words[i].append(model.id2token[idx])
    return cluster_words

''' PLOTS  '''

def mask_(url="https://image.shutterstock.com/image-illustration/flask-word-cloud-artwork-isolated-260nw-185529119.jpg"):
    return np.array(
        Image.open(
            requests.get(url,stream=True).raw))

def clusters_to_wordclouds(
    tfidf_df:pd.DataFrame, 
    cluster_words,
    use_mask=True,
    n_cols=3):
    '''
    Create a grid with a WordCloud for each Cluster
    '''
    n_rows = len(cluster_words)//n_cols
    _, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols*5,n_rows*5))
    # For each cluster
    for cluster, words in cluster_words.items():
        # Filter the tfidf to the words in this cluster
        subtfidf = tfidf_df[tfidf_df.word.isin(words)]
        # Create the wordcloud attending to the inverse of idf
        wordcloud = WordCloud(
            max_words=100, 
            mask=mask_ if use_mask else None,
            background_color="white").generate_from_frequencies(
                frequencies=dict(zip(subtfidf.word, subtfidf.idf)))
        # Plot the resulting wordcloud
        axs[cluster // n_cols, cluster % n_cols].imshow(wordcloud)
        axs[cluster // n_cols, cluster % n_cols].axis('off')
    plt.tight_layout()
    plt.show()
    return



'''
HIERARCHICAL CLUSTERING 
=======================
'''

def tfidf_to_idf_dict(model):
    return {
        "word": [ k for k,v in model.token2id.items() ],
        "idf":  [ model.mapping.idf_[v] for k,v in model.token2id.items()]}


def tfidf_to_idf_scores(model): # Model
    d = tfidf_to_idf_dict(model)
    return pd.DataFrame().sort_values("idf",ascending=False)


def get_most_relevant_terms(
    tfidf_df:pd.DataFrame,
    n_terms:int):
    ''' Return the first max_terms terms relevant by their IDF value '''
    if not isinstance(tfidf_df,pd.DataFrame):
        tfidf_df = tfidf_to_idf_scores(tfidf_df)
    return tfidf_df.sort_values(
        by='idf', ascending=False).iloc[:n_terms,:]['word'].tolist()


def subsample_by_idf(model):
    return


def ward_clustering(
    model, # Model
    tfidf_df:pd.DataFrame=None,
    n_terms:int=None):
    '''
    Performs Ward Hierarchical Cluster
    Arguments:
        - model: Model instance representation
        - tfidf_df: pd.DataFrame of the IDF scores for the terms of that model
        - max_terms: filter first max_terms terms to use for the cluster by idf
    NOTE: 
        Filtering by terms is breaking the whole thing when plotting ??
        Does it make sense to run it on all and then truncate the plot rather than
        subsampling the TFIDF by the most important words (columns)?
    '''
    # If not DF representation of the model is passed, compute it
    if tfidf_df is None:
        tfidf_df = tfidf_to_idf_scores(model)
    # If not list of most relevant terms by IDF is passed, compute it
    if n_terms is None:
        n_terms = tfidf_df.shape[1]    
    
    terms = get_most_relevant_terms(tfidf_df, n_terms)
    X = model.representation

    ## NOTE: Finally found something weird going on here.. Needs transposition
    X = X[terms].T
    print('Shape of filtered TFIDF Matrix by IDF: ', X.shape)

    dist = 1 - cosine_similarity(X)
    linkage_matrix = ward(dist)
    print('Shape of resulting Linkage Matrix', linkage_matrix.shape)
    return linkage_matrix


def plot_dendogram_from_catalog(
    model, 
    n_terms:int,
    truncate_mode=None,
    clusters:int=5):
    linkage_matrix = ward_clustering(model=model,n_terms=n_terms)
    terms = get_most_relevant_terms(tfidf_df=model,n_terms=n_terms)
    plot_dendogram_from_linkage_matrix(
        linkage_matrix=linkage_matrix,
        truncate_mode=truncate_mode,
        clusters=clusters,
        labels=terms)
    return


def plot_dendogram_from_linkage_matrix(
    linkage_matrix, 
    truncate_mode=None,
    clusters:int=None,
    labels:list=None,
    orientation='right'):
    ''' Plot a dendogram out of its linkage matrix '''
    
    _, ax = plt.subplots(figsize=(15, 20)) # set size

    dendrogram(
        Z=linkage_matrix,
        p=clusters,                   # p == clusters 
        truncate_mode=truncate_mode,  # show only the last p merged clusters
        orientation=orientation, 
        labels=labels, 
        ax=ax)

    plt.tick_params(
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout
    plt.show()
    return


if __name__ == '__main__':

    '''
    PIPELINE UNTIL THE TFIDF
    ------------------------
    '''    
    import sys
    sys.path.append('../utils')
    sys.path.append('../scripts')
    from scripts.catalog import load_corpus, Catalog
    from sklearn.feature_extraction.text import TfidfVectorizer

    catalog = Catalog()
    corpus = load_corpus(path=paths['catalog'], name='corpus1')
    catalog.load_corpus(corpus=corpus)

    filters = dict(
        topic = ['isocyanate'],
        label='relevant',
        country = ['CN'],       # country = OF_INTEREST
        raw_text_len = 100)

    pos_catalog = catalog.filter_catalog(filters)
    pos_catalog.collect_corpus()

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
        stop_words=SW)

    pos_tfidf = pos_catalog.to_matrix(
        vectorizer=vectorizer,
        modelname='TFIDF',
        max_docs=50)

    pos_tfidf.representation.head()
    pos_tfidf_df = tfidf_to_dataframe(pos_tfidf)
    pos_tfidf_df.head()

    pos_terms = get_most_relevant_terms(pos_tfidf_df,max_terms=50)
    

    ''' REGUALAR CLUSTERING '''
    NUM_CLUSTERS = 6
    WORDS_PER_CLUSTER = 500

    clustered_words = kmean_clustering(
        model=pos_catalog.models['TFIDF'],
        num_clusters=NUM_CLUSTERS, 
        words_per_cluster=WORDS_PER_CLUSTER)

    ''' Clustering2WordCloud '''
    clusters_to_wordclouds(pos_tfidf_df, clustered_words)


    ''' HIERARCHICAL CLUSTERING '''
    # Alternative 1 - Computing everything in advanced
    MAX_TERMS = 500
    pos_terms = pos_tfidf_df.sort_values(
        by='idf',ascending=False).iloc[:MAX_TERMS,:]['word'].tolist()

    X = pos_catalog.models['TFIDF'].representation
    X = X[pos_terms]
    X.head(5)

    dist = 1 - cosine_similarity(X)
    linkage_matrix = ward(dist)
    plot_dendogram_from_linkage_matrix(linkage_matrix, pos_terms)



    # Aternative 2 - Directly from the catalog.models['TFIDF']
    pos_matrix = ward_clustering(
        model=pos_catalog.models['TFIDF'],
        tfidf_df=pos_tfidf_df,
        terms = pos_terms)

    ''' Clustering to Dendogram ''' 
    plot_dendogram_from_linkage_matrix(pos_matrix, clusters=5)
    # plot_dendogram_from_linkage_matrix(pos_catalog.models['TFIDF'])
