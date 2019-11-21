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
from scripts.catalog import Catalog, load_catalog

import spacy
from spacy import displacy

config = parse_yaml('config.yaml')
paths = config['paths']
catalog = load_catalog(path=paths['catalog'],name='only_US')
print(len(catalog.documents))

docu1 = catalog.documents[0]
docu2 = catalog.documents[1]
raw_text = docu1.raw_text
text = docu1.clean_text

text[250:500]

'''
SPACY PIPELINE
--------------
'''
nlp = spacy.load('en_core_web_sm') # Powerfull model with everytihing included
d = nlp(text)

displacy.render(d[250:500],style='ent',jupyter=True)

print(d[1].text)
print(d[1].lemma_)
print(d[1].tag_)

def spacy_cleaning(
    document,
    tags_to_keep=['JJ', 'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    entities_to_remove=['ORG,NORP,GPE,PERSON']):

    def pass_test(w, tags=tags_to_keep):
        if w.ent_type_ == 0:
                return w.tag_ in tags and not w.is_punct and not w.is_stop and w.ent_ not in entities_to_remove
        return w.tag_ in tags and not w.is_punct and not w.is_stop 

    words = [ word for word in d if pass_test(word)]
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in words ]
    return ' '.join(tokens)

tokens = spacy_cleaning(d)
tokens[250:500]
print(tokens[:1000])