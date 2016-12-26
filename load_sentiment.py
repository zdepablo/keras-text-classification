
def load_sentiment_file(filename):
    f = open(filename)
    X = list()
    for line in f:
        X.append(line.strip().split(" "))
    return X

def load_sentiment_corpus(filepath):
    pos_filename =  dirpath + "rt-polarity.pos"
    neg_filename =  dirpath + "rt-polarity.neg"
    
    X_pos = load_sentiment_file(pos_filename)
    X_neg = load_sentiment_file(neg_filename) 
    
    Y_pos = ["pos"] * len(X_pos)
    Y_neg = ["neg"] * len(X_neg)
    
    X = X_pos + X_neg
    Y = Y_pos + Y_neg
    
    return (X,Y)

def load_subjectivity_corpus(filepath):
    quote_filename = dirpath + "quote.tok.gt9.5000"
    plot_filename  = dirpath + "plot.tok.gt9.5000"
    
    X_quote = load_sentiment_file(quote_filename)
    X_plot  = load_sentiment_file(plot_filename)
    
    Y_quote = ["quote"] * len(X_quote)
    Y_plot =  ["plot"]  * len(X_plot)

    X = X_quote + X_plot
    Y = Y_quote + Y_plot
    
    return (X,Y)