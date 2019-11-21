
import os
import json
import base64
from glob import glob
from pprint import pprint
from os.path import join as JP

from utils.nlp_utils import preproces
from utils.general import parse_yaml, ensure_directories
from scripts.catalog import Catalog, Document, load_catalog

from utils.datahandling import (
    filter_dict_by_keys, 
    filter_dict_by_vals,
    filter_df_mean_thres)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

config = parse_yaml('config.yaml')
paths = config['paths']
ensure_directories(paths)

DATA_PATH = paths['data']
TOPIC = config['topic']
CLASSES = config['classes']

catalog = load_catalog(path=paths['catalog'], name='test1')
catalog.documents[TOPIC]
print('Loaded total of {} documents for topic "{}"'.format((len(catalog.documents[TOPIC])), TOPIC))

# Apply first cleaninig logic -- TODO: Improve upon this clearning
# for document in catalog.documents[TOPIC]:
#     document.clean_document()
# catalog.save(path=paths['catalog'],name='test1') ----> Save to avoid hard task


# HOW MANY DOCUMENT BY COUNTRY
# ----------------------------
ndocs = defaultdict(int)
for document in catalog.documents[TOPIC]:
    ndocs[document.country] += 1
    
df_ndocs = pd.DataFrame.from_dict(
    dict([(k,pd.Series(v)) for k,v in ndocs.items()]))
    
plt.figure(figsize=(15,5))
plt.bar(ndocs.keys(),ndocs.values())
plt.title('Number of Docs per Country')
plt.show()

plt.figure(figsize=(15,5))
plt.bar(filter_dict_by_keys(ndocs,'CN').keys(),filter_dict_by_keys(ndocs,'CN').values())
plt.title('Number of Docs per Country without China')
plt.show()

OF_INTEREST = list(filter_dict_by_vals(ndocs,low_thres=100).keys())
print(OF_INTEREST)



# LENGTH OF ALL DOCUMENTS
# -----------------------
ldocs = list()
for document in catalog.documents[TOPIC]:
    ldocs.append(len(document.clean_text))
plt.figure(figsize=(15,3))
plt.hist(ldocs,bins=80)
plt.title('Histogram of Lenght of Documents')
plt.show()


# LENGTH OF ALL DOCUMENTS
# -----------------------
ls = list()
ls2 = list()
for document in catalog.documents[TOPIC]:
    l = len(document.clean_text)
    ls.append(l)
    if document.country != 'CN':
        ls2.append(l)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.hist(ls,bins=200)
ax2.hist(ls2,bins=200)
ax1.set_xlim(0,np.max(ls)*0.3)
ax2.set_xlim(0,np.max(ls)*0.3)
ax1.set_title('Distribution of Lenght of Documents')
ax2.set_title('Distribution of Lenght of Documents without China')
plt.tight_layout()
plt.show()



# EXPLORE LENGHT OF DOCUMENTS BY COUNTRY
# --------------------------------------

from collections import defaultdict
lengths = defaultdict(list)
for document in catalog.documents[TOPIC]:
    lengths[document.country].append(len(document.clean_text))
df_lengths = pd.DataFrame.from_dict(
    dict([(k,pd.Series(v)) for k,v in lengths.items()]))


# 1 - Average lenght of text by Country
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
plt.title('Average Lenght of Text by Country')
df_lengths.mean(axis=0).plot('bar', ax=ax)
plt.show()


data = pd.merge(df_ndocs.T,pd.DataFrame(df_lengths.mean()),left_index=True, right_index=True)
df = pd.DataFrame(data.values, columns = ['AverageLength', 'NumberOfDocuments'], index=data.index)
df = df.T[OF_INTEREST].T

fig = plt.figure(figsize=(15,5))
plt.title('Average Lenght of Text and Number of Documents by Country')
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
df.AverageLength.plot('bar', ax=ax1, color='blue', width=0.4, position=0)
df.NumberOfDocuments.plot('bar', ax=ax2, color='red', width=0.4, position=1)
plt.show()




# Let's filter those with at least an average above 20.000
LOW_TH = 20000
UPP_TH = 50000
def df_mean_thres(df,low_thres=None, upp_thres=None):
    if not low_thres and upp_thres:
        return df.loc[:,df.mean(axis=0) < upp_thres]     
    if low_thres and not upp_thres:
        return df.loc[:,df.mean(axis=0) > low_thres] 
    # return df.loc[:, df.mean(axis=0).between(low_thres, upp_thres)]
    return df.loc[:,(df.mean(axis=0) > low_thres) & (df.mean(axis=0) < upp_thres)] 

# 2 - Distibution of lenght of text by Country
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(111)
plt.title('Average Lenght of Text by Country')
df_mean_thres(df_lengths,
    low_thres=LOW_TH, upp_thres=None).plot(
        kind='hist',bins=10*len(list(df_mean_thres(df_lengths,LOW_TH))), alpha=.5, ax=ax)
ax.set_xlim(0,np.max(df_lengths.mean(axis=0)*0.7))
plt.show()




# texts = list()
# for document in catalog.documents[TOPIC]:
#     texts.append(document.process_document())


# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import RegexpTokenizer
# #sentences = [preproces(s) for s in docnorm_text.split('\n')]
# sentences = [s for s in docnorm_text.split('\n') if check_sent(s)]
# sentences = sent_tokenize(docnorm_text)
# sentences[0:10]