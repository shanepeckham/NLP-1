
import os
import nltk
import random
from pprint import pprint

# text = '''
# Global warming or climate change has become a worldwide concern. It is gradually developing into an unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due to entrapment of greenhouse gases in the earth’s atmosphere."""
# '''

os.chdir('/Users/pabloruizruiz/OneDrive/Courses/NLP Stanford Course/')
print(os.getcwd())

with open('Courses/NLTK/dicaprio.txt') as f:
    text = f.read()


# WORD-BASED N-GRAN MODEL
# -----------------------

# Create Model

N = 7
ngrams = dict()
words = nltk.word_tokenize(text)

for i in range(len(words)-N):
    # key - values
    gram = ' '.join(words[i:i+N])
    value = words[i+N]
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(value)

# pprint(ngrams)

# Evaluate Model
# --------------

curGram = ' '.join(words[0:N])
result = curGram

for _ in range(300):
    if curGram not in ngrams.keys():
        print('Not found for: ', curGram)
        break
    possibilities = ngrams[curGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += ' '+nextItem
    rwords = nltk.word_tokenize(result)
    curGram = ' '.join(rwords[len(rwords)-N:len(rwords)])

print(result)
