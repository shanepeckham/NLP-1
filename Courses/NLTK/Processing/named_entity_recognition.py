

import os
import re
import nltk

nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

paragraph = 'The Taj Mahal was built by Emperor Shah Jahan'

"""
ORGANIZATION	Georgia-Pacific Corp., WHO
PERSON	        Eddy Bonte, President Obama
LOCATION	    Murray River, Mount Everest
DATE	        June, 2008-06-29
TIME	        two fifty a m, 1:30 p.m.
MONEY	        175 million Canadian Dollars, GBP 10.40
PERCENT	        twenty pct, 18.75 %
FACILITY	    Washington Monument, Stonehenge
GPE	            South East Asia, Midlothian
"""

words = nltk.word_tokenize(paragraph)
tagged_words = nltk.pos_tag(words)

entities = nltk.ne_chunk(tagged_words)
entities.draw()

