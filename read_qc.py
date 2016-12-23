## Question Classification Dataset
## http://cogcomp.cs.illinois.edu/Data/QA/QC/

def split_question(question):
    q = question.strip().split(" ")
    return (q[0],q[1:])

split_question("LOC:other What is the longest suspension bridge in the U.S. ?\n")

def load_question_file(filename):
    f = open(filename)
    X = list()
    Y = list()
    for line in f:
        (y,x) = split_question(line)
        Y.append(y)
        X.append(x)
    return (Y,X)
    

path = "/home/ec2-user/data/qc/"
(Y_train,X_train) = load_question_file(path + "train_5500.label")
(Y_test,X_test) = load_question_file(path + "TREC_10.label")
