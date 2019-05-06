import re

import pandas as pd
import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras_preprocessing.sequence import pad_sequences
import keras
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout
from keras.layers.embeddings import Embedding
from load_glove_embeddings import load_glove_embeddings
import pickle
import matplotlib.pyplot as plt

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
    # word2index, embedding_matrix = load_glove_embeddings('/home/federica/Documents/Thesis/Geras/Software/texW2V/wiki-news-300d-1M.vec', embedding_dim=50)
    #
    # out_matrix = []
    #
    # for text in x['text'].tolist():
    #     indices = []
    #     for w in text_to_word_sequence(text):
    #         indices.append(word2index[re.sub(r'[^\w\s]', '', w)])
    #
    #     out_matrix.append(indices)
    #
    # encoded_texts = out_matrix
    #
    # max_len = 70
    # padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')
    #
    # # for idx, el in enumerate(padded_texts):
    # #     a = np.array(el)
    # #     a = np.reshape(a, (70, 1))
    # #     padded_texts[idx] = a

    # # for idx, el in enumerate(df['text']):
    # #     if df['text'][idx] != 'text':
    # #         a = df['text'][idx]
    # #         a = np.reshape(a, (70, 1))
    #         x[idx] = a
    # store_data(padded_texts, 'embedded_text')
    padded_texts = load_data('embedded_text')
    # store_data(embedding_matrix, 'embedded_matrix')
    embedding_matrix = load_data('embedded_matrix')
    for idx, el in enumerate(padded_texts):
        df['text'][idx] = el
    do_kfold_validation(x, y, embedding_matrix, padded_texts)


def store_data(padded_texts, file_name):
    pickle_file = open(file_name, 'ab')

    pickle.dump(padded_texts, pickle_file)
    pickle_file.close()


def load_data(file_name):
    pickle_file = open(file_name, 'rb')
    padded_texts = pickle.load(pickle_file)

    pickle_file.close()
    return padded_texts


"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation(x, y, embedding_matrix, padded_texts):
    # initializes kfold with 5 folds, including shuffling,
    # using 7 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=7, shuffle=True)

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(df['text']):
        x_train, x_val = df['text'].loc[train_index], df['text'].loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]

    # for train_index, val_index in kfold.split(padded_texts):
    #     x_train, x_val = padded_texts[train_index:], padded_texts[:val_index]
    #     y_train, y_val = y.loc[train_index], y.loc[val_index]

    train_model(x_train, y_train, embedding_matrix, 70, x_val, y_val, padded_texts, y)


def train_model(x_train, y_train, embedding_matrix, maxlen, x_val, y_val, padded_texts, y):

    # for idx, el in enumerate(x_train):
    #     if el == 'text':
    #         x_train.pop(0)
    #     a = el
    #     a = np.reshape(a, (70, 1))
    #     x_train[idx] = a

    # x_train.pop(0)


    x_df = x_train.values
    y_df = x_val.values

    # a = np.zeros([353, 70])
    a = np.vstack(x_df)

    # b = np.zeros([88, 70])
    b = np.vstack(y_df)

    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False,
                                name='embedding_layer'))
    model.add(LSTM(128, return_sequences=True, dropout=0.2))
    model.add(LSTM(128, dropout=0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
    print(model.summary())  # Summarize the model
    es = EarlyStopping(monitor='val_acc', patience=30)
    history = model.fit(a, y_train, epochs=400,  verbose=1, validation_data=(b, y_val), batch_size=a.shape[0])  # Fit the model
    # history = model.fit(padded_texts, y, batch_size=padded_texts.shape[0], epochs=3000, verbose=1, callbacks=[es])  # Fit the model
    # loss, accuracy = model.evaluate(a, y_train, verbose=0)  # Evaluate the model
    # print('Train set accuracy: %0.3f' % accuracy)
    # loss, accuracy = model.evaluate(b, y_val, verbose=0)
    # print('Validation set accuracy: %0.3f' % accuracy)
    plot_history(history)


"""
Evaluates accuracy of the model
"""


def evaluate_model(model, history, x_train, y_train):
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', scores[1] * 100))
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', float(history.history['val_acc'][-1]) * 100))


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss


df = read_csv()
split_dataframe(df)
# do_kfold_validation()
