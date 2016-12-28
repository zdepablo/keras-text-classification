
# scikit-learn 0.17
from sklearn.cross_validation import train_test_split
from text_utils import build_dict, generate_sequence

# scikit-learn 0.18 
# from sklearn.model_selection import train_test_split

def load_sentiment_file(filename):
    f = open(filename)
    X = list()
    for line in f:
        X.append(line.strip().split(" "))
    return X

def load_sentiment_corpus(dirpath):
    pos_filename =  dirpath + "rt-polarity.pos"
    neg_filename =  dirpath + "rt-polarity.neg"
    
    X_pos = load_sentiment_file(pos_filename)
    X_neg = load_sentiment_file(neg_filename) 
    
    Y_pos = ["pos"] * len(X_pos)
    Y_neg = ["neg"] * len(X_neg)
    
    X = X_pos + X_neg
    Y = Y_pos + Y_neg
    
    return (X,Y)

def load_subjectivity_corpus(dirpath):
    quote_filename = dirpath + "quote.tok.gt9.5000"
    plot_filename  = dirpath + "plot.tok.gt9.5000"
    
    X_quote = load_sentiment_file(quote_filename)
    X_plot  = load_sentiment_file(plot_filename)
    
    Y_quote = ["quote"] * len(X_quote)
    Y_plot =  ["plot"]  * len(X_plot)

    X = X_quote + X_plot
    Y = Y_quote + Y_plot
    
    return (X,Y)

def load_sentiment_data(dirpath, max_tokens = 0):
    (X,Y) = load_sentiment_corpus(dirpath)
    X_train_sentences, X_test_sentences, y_train_label, y_test_label = train_test_split(X,Y, test_size = 0.1, random_state = 43)
    
    worddict, wordcount = build_dict(X_train_sentences)
    
    if (max_tokens > 0 ):
        filterdict = {k:v for k,v in worddict.iteritems() if v < max_tokens } 
    else: 
        filterdict = worddict
        
    print 'Filtering to', (len(filterdict) + 2) , ' unique words'
    
    X_train = generate_sequence(X_train_sentences, filterdict)
    X_test  = generate_sequence(X_test_sentences, filterdict)
    
    labels = set(y_train_label + y_test_label)
    catdict = {label: idx for (idx, label) in enumerate(labels)}
    
    y_train = [catdict[y] for y in y_train_label]
    y_test  = [catdict[y] for y in y_test_label]
    
    return X_train, X_test, y_train, y_test, worddict