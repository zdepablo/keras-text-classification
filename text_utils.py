
import numpy as np

def build_dict(sentences):
#    from collections import OrderedDict

    '''
    Build dictionary of train words
    Outputs: 
     - Dictionary of word --> word index
     - Dictionary of word --> word count freq
    '''
    print 'Building dictionary..',
    wordcount = dict()
    #For each worn in each sentence, cummulate frequency
    for ss in sentences:
        for w in ss:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values() # List of frequencies
    keys = wordcount.keys() #List of words
    
    sorted_idx = reversed(np.argsort(counts))
    
    worddict = dict()
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)
    print np.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict, wordcount

def generate_sequence(sentences, dictionary):
    '''
    Convert tokenized text in sequences of integers
    '''
    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]

    return seqs