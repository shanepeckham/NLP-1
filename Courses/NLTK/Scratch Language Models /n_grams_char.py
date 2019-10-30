
import random
from pprint import pprint

text = '''
Global warming or climate change has become a worldwide concern. It is gradually developing into an unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due to entrapment of greenhouse gases in the earth’s atmosphere."""
'''


# CHAR-BASED N-GRAN MODEL
# -----------------------

# Create Model

N = 3
ngrams = dict()

for i in range(len(text)-N):
    # key - values
    gram = text[i:i+N]
    value = text[i+N]
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(value)

pprint(ngrams)

# Evaluate Model --> Generate text and see

currentGram = text[0:N]
result = currentGram

verbose = True
for i in range(100):
    if currentGram not in ngrams.keys():
        break
    possibilites = ngrams[currentGram]
    if verbose: print('Gram: {} || Possibilities: {}'.format(currentGram, possibilites))
    nextItem = possibilites[random.randrange(len(possibilites))]
    result += nextItem
    currentGram = result[len(result)-N:len(result)]

print(result)


def train_and_run(N, verbose=True):
    
    ngrams = dict()

    for i in range(len(text)-N):
        # key - values
        gram = text[i:i+N]
        value = text[i+N]
        if gram not in ngrams.keys():
            ngrams[gram] = []
        ngrams[gram].append(value)

    currentGram = text[0:N]
    result = currentGram

    for i in range(100):
        if currentGram not in ngrams.keys():
            break
        possibilites = ngrams[currentGram]
        if verbose: print('Gram: {} || Possibilities: {}'.format(currentGram, possibilites))
        nextItem = possibilites[random.randrange(len(possibilites))]
        result += nextItem
        currentGram = result[len(result)-N:len(result)]

    print(result)


train_and_run(5)