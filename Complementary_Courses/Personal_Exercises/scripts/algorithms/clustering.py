

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
UTILS
=====
'''
# def tfidf_to_dataframe(model): # Model
#     ''' Given a Model instance, returns its IDF score for each word as a DataFrame '''
#     return pd.DataFrame({
#         "word": [ k for k,v in model.token2id.items() ],
#         "idf":  [ model.mapping.idf_[v] \
#                     for k,v in model.token2id.items()]
#         }).sort_values("idf",ascending=False)

# def get_most_relevant_terms(
#     tfidf_df:pd.DataFrame,
#     n_terms:int):
#     ''' Return the first max_terms terms relevant by their IDF value '''
#     if not isinstance(tfidf_df,pd.DataFrame):
#         tfidf_df = tfidf_to_dataframe(tfidf_df)
#     return tfidf_df.sort_values(
#         by='idf', ascending=False).iloc[:n_terms,:]['word'].tolist()

def subsample_tfidf_by_cluster(model,words):
    ''' Return a sample version of a TFIDF for the words of a cluster '''
    return model[words]

def get_tf_idf_of_word_from_tfidf_matrix(model,k,v):
    ''' Get the TF vector and IDF float of an specific term '''
    return model.representation[[k]].values, model.mapping.idf_[v]

def compute_word_importance(model,words_of_cluster=None):
    ''' Compute the importance of a word given the TFIDF for a bunch of
    importance methods '''
    scores = defaultdict(list)
    if not words_of_cluster: words_of_cluster = model.token2id.keys()
    for k,v in model.token2id.items():
        # With this comparison we save a lot of time
        if k in words_of_cluster:
            scores['words'].append(k)
            t,i = get_tf_idf_of_word_from_tfidf_matrix(model,k,v)
            scores['idf'].append(i)
            scores['max_tf_idf'].append(np.max(t)*i)
            scores['avg_tf_idf'].append(np.max(t)*i)
            scores['norm_tf_idf'].append(np.linalg.norm(t)*i)
    return scores

# def gather_clusters_information(model,cluster_words):
#     ''' 
#     Construct a dictionary where the key is the cluster ID 
#     and the values are all the information computed about it:
#         - Most important words
#         - TFIDF representation of the cluster for those words
#         - Scores for the different methods for each of those words
#     '''
#     clusters = defaultdict(dict)
#     for c,words in cluster_words.items():
#         ws = {'words': words}
#         tfidf = subsample_by_idf(model,words)
#         scores = compute_word_importance



'''
REGULAR CLUSTERING 
==================
'''

''' ALGORITHM '''

def kmean_clustering(
    model, #:Model
    num_clusters:int=4, 
    words_per_cluster:int=None):
    '''
    TODO: Consider MiniBatchKMeans
    
    Clusters using k-mean with k words per cluster
    ----------------------------------------------
        The k-words are the k closest to the centroid of that cluster
        Equivalently: the words are the ones most present in the 'fake'
        document represented by the centroid of the cluster

    Inputs:
    -------
        - model: Trained instance of class Model
        - num_clusters: Number of Clusters to look for
        - words_per_cluster: K parameter above

    Returns:
    -------- 
        - Dict key='cluster id', value=k_words_closest_to_centroid
    '''
    # 1. Performs K-Means algorithm to identify clusters
    km = KMeans(
        n_clusters=num_clusters,
        n_jobs=-1)
    km.fit_transform(model.representation)
    # clusters = km.labels_.tolist()

    # Bring K most similar words to centroid
    closests_words_to_centroids = km.cluster_centers_.argsort()[:, :-words_per_cluster:-1] 
    
    cluster_words = defaultdict(list)
    for i in range(num_clusters):
        for idx in closests_words_to_centroids[i, :words_per_cluster]:
            cluster_words[i].append(model.id2token[idx])
    return cluster_words



''' PLOTS  '''

def mask_(url="https://image.shutterstock.com/image-illustration/flask-word-cloud-artwork-isolated-260nw-185529119.jpg"):
    return np.array(
        Image.open(
            requests.get(url,stream=True).raw))


def cluster_to_wordcloud(
    df, method='idf', max_words=100, use_mask=False):
    ''' Convert 1 cluster into a WordCloud given:
        - The TFIDF for the cluster
        - The Score Method that give imporance to the word '''
    # Create the wordcloud attending to the inverse of idf
    wordcloud = WordCloud(
        max_words=max_words, 
        mask=mask_ if use_mask else None,
        background_color="white").generate_from_frequencies(
            frequencies=dict(zip(df.words, df[method])))
    return wordcloud



def plot_clusters_as_wordclouds(
    tfidf:pd.DataFrame, 
    cluster_words:dict,
    method:str='idf',
    max_words_per_cloud=100, use_mask=False, n_cols=2):
    '''
    Arguments:
        - tfidf: TFIDF of the entire Corpus
        - cluster_words: Dict {'cluster_id': [list of important words]}
        - methods: the Score Method that give imporance to the word in that cluster
    Steps:
        - Iterate for each cluster
        - Subsample the TFIDF to the Cluster TDIDF (reduce the columns to increase performance)
        - Get the scores of the chosen methods to give importance to the words --> Not needed ???
        - Call cluster_to_wordcloud() for that cluster to get its corresponding wordcloud
    '''
    n_rows = len(cluster_words)//n_cols
    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(n_cols*5,n_rows*5))
    
    for cluster,words in cluster_words.items():

        cluster_word_scores = pd.DataFrame(compute_word_importance(tfidf,cluster))
        print(cluster_word_scores.head())
        print(cluster_word_scores.shape)
        # cluster_tfidf = sort_scores(scores, 'norm_tf_idf')
        wordcloud = cluster_to_wordcloud(
            df=cluster_word_scores,
            method='norm_tf_idf',
            max_words=max_words_per_cloud,
            use_mask=use_mask)
        
        # Plot the resulting wordcloud
        axs[cluster // n_cols, cluster % n_cols].imshow(wordcloud)
        axs[cluster // n_cols, cluster % n_cols].axis('off')
    plt.tight_layout()
    plt.show()
    return
    

# def clusters_to_wordclouds(
#     tfidf_df:pd.DataFrame, 
#     cluster_words,
#     use_mask=True,
#     n_cols=3):
#     '''
#     Create a grid with a WordCloud for each Cluster
#     '''
#     n_rows = len(cluster_words)//n_cols
#     _, axs = plt.subplots(
#         nrows=n_rows, ncols=n_cols,
#         figsize=(n_cols*5,n_rows*5))
#     # For each cluster
#     for cluster, words in cluster_words.items():
#         # Filter the tfidf to the words in this cluster
#         subtfidf = tfidf_df[tfidf_df.word.isin(words)]
#         # Create the wordcloud attending to the inverse of idf
#         wordcloud = WordCloud(
#             max_words=100, 
#             mask=mask_ if use_mask else None,
#             background_color="white").generate_from_frequencies(
#                 frequencies=dict(zip(subtfidf.word, subtfidf.idf)))
#         # Plot the resulting wordcloud
#         axs[cluster // n_cols, cluster % n_cols].imshow(wordcloud)
#         axs[cluster // n_cols, cluster % n_cols].axis('off')
#     plt.tight_layout()
#     plt.show()
#     return







'''
HIERARCHICAL CLUSTERING 
=======================
'''

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
