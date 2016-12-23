
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Input, Bidirectional

def build_lstm(max_features, embedding_dims, nb_classes):  
   model = Sequential()

   model.add(Embedding(max_features, 128, dropout=0.2))
   model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
   model.add(Dense(nb_classes))
   model.add(Activation('softmax'))

   return model


def build_cnn(embedding_dims, maxlen, nb_filter, filter_length, hidden_dims, nb_classes ) : 
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
    # we use max pooling:
    model.add(MaxPooling1D(pool_length=model.output_shape[1]))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    return model


def build_cnn_lstm(embedding_size, maxlen, nb_filter, filter_length, pool_length, lstm_output_size, nb_classes):
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
     
    return model     

def build_bidirectional_lstm(embedding_dims, lstm_output_size, nb_classes):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Bidirectional(LSTM(lstm_output_size)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model