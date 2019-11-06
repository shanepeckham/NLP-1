
'''
Process Text from multiple sources and formats.
Convert data between plain Python, Pandas for Scikit-Learn and Gensim Data Structures

'''

import os
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from os.path import dirname
os.chdir(dirname(__file__))

