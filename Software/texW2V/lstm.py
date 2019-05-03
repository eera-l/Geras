import re

import pandas as pd
import numpy as np
from keras import Input, Model
from keras_preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras_preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.layers.embeddings import Embedding
from load_glove_embeddings import load_glove_embeddings

"""
Reads data from .csv file
"""


def read_csv():

    # reads data and stores it into dataframe
    df = pd.read_csv('train_set_lstm', sep=',')
    return df


"""
Splits dataframe into x (containing the data)
and y (containing the respective labels)
"""


def split_dataframe(df):

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia'])
    embed(x, df, y)


def embed(x, df, y):
    word2index, embedding_matrix = load_glove_embeddings('wiki-news-300d-1M.vec', embedding_dim=50)

    out_matrix = []

    for text in x['text'].tolist():
        indices = []
        for w in text_to_word_sequence(text):
            indices.append(word2index[re.sub(r'[^\w\s]', '', w)])

        out_matrix.append(indices)

    encoded_texts = out_matrix

    max_len = 50
    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')
    # print(padded_texts)
    df['text'] = padded_texts
    x = df.drop(columns=['dementia'])
    do_kfold_validation(x, y, embedding_matrix, padded_texts)



"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation(x, y, embedding_matrix, padded_texts):
    # initializes kfold with 5 folds, including shuffling,
    # using 7 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=7, shuffle=True)

    # x_train, y_train = []
    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]

    train_model(x_train, y_train, embedding_matrix, 50, padded_texts)


def train_model(x_train, y_train, embedding_matrix, maxlen, padded_texts):

    # embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
    #                             output_dim=embedding_matrix.shape[1],
    #                             input_length=maxlen,
    #                             weights=[embedding_matrix],
    #                             trainable=False,
    #                             name='embedding_layer')

    # i = Input(shape=(maxlen,), dtype='int32', name='main_input')
    # x = embedding_layer(i)
    # x = Flatten()(x)
    # # x_train = x
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False,
                                name='embedding_layer'))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
    print(model.summary())  # Summarize the model
    model.fit(padded_texts, y_train, batch_size=147, epochs=50, verbose=0)  # Fit the model
    loss, accuracy = model.evaluate(padded_texts, y_train, verbose=0)  # Evaluate the model
    print('Accuracy: %0.3f' % accuracy)


"""
Evaluates accuracy of the model
"""


def evaluate_model(model, history, x_train, y_train):
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', scores[1] * 100))
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', float(history.history['val_acc'][-1]) * 100))


df = read_csv()
split_dataframe(df)
# do_kfold_validation()
