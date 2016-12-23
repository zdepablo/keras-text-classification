## Question Classification Dataset
## http://cogcomp.cs.illinois.edu/Data/QA/QC/

import numpy as np

def split_question(question):
    q = question.strip().split(" ")
    return (q[0],q[1:])
file

def load_question_file(filename):
    f = open(filename)
    X = list()
    Y = list()
    for line in f:
        (y,x) = split_question(line)
        Y.append(y)
        X.append(x)
    return (Y,X)

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

def parse_label(label):
    t = label.split(":")
    return (t[0],t[1])

def load_corpus(path): 
    (Y_train_full,X_train_sentences) = load_question_file(path + "train_5500.label")
    (Y_test_full,X_test_sentences) = load_question_file(path + "TREC_10.label")
    return (Y_train_full,X_train_sentences), (Y_test_full,X_test_sentences)

def load_data(path): 
    (Y_train_full,X_train_sentences), (Y_test_full,X_test_sentences) = load_corpus(path)
    worddict, wordcount = build_dict(X_train_sentences)
    
    X_train = generate_sequence(X_train_sentences, worddict)
    X_test  = generate_sequence(X_test_sentences, worddict)
    
    Y_train_label = [parse_label(y)[0]  for y in Y_train_full]
    Y_test_label  = [parse_label(y)[0]  for y in Y_test_full]
    
    labels = set(Y_train_label + Y_test_label)
    catdict = {label: idx for (idx, label) in enumerate(labels)}
    
    Y_train = [catdict[y] for y in Y_train_label]
    Y_test  = [catdict[y] for y in Y_test_label]
    
    return (Y_train,X_train), (Y_test,X_test)