
import numpy as np

# Candidate Model
def edits_1(word):
    '''
    All edits that are 1 edit away from `word`.
    Regardless if the word exists or not --> This generate too many candidates
    It will smarter if we just look for known words in the dictionary
    ''' 
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return splits, deletes, transposes, replaces, inserts
    #return set(deletes + transposes + replaces + inserts)

def edits_2(word): 
    '''
    All edits that are 2 edits away.
    This generates more possibilites but only few od them will be known words.
    '''
    return (e2 for e1 in edits_1(word) for e2 in edits1(e1))


def known(words): 
    return set(w for w in words if w in WORDS)


splits, deletes, transposes, replaces, inserts = edits_1('thew')



