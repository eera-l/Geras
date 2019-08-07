"""
lstm.py
author: Federica Comuni
Implements an LSTM. Optionally performs embedding of the text input.
Reads features and test from input files, splits dataset into data and labels,
trains and evaluates model.
"""
import re

import pandas as pd
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, concatenate, \
    BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import StandardScaler

from load_embeddings import load_embeddings
import pickle
import matplotlib.pyplot as plt


def read_csv():
    """
    Reads data from .csv file
    :return: 4 dataframes: training features, training text,
    test features, test text
    """
    # reads data and stores it into dataframe
    df = pd.read_csv('train_set_lstm', sep=',')
    df_2 = pd.read_csv('train_set', sep=',')
    df_test = pd.read_csv('test_set', sep=',')
    df_test_lstm = pd.read_csv('test_set_lstm', sep=',')
    dataframes = [df, df_2, df_test, df_test_lstm]
    np.random.seed(21)
    return dataframes


def split_dataframe(dataframes, needs_embed):
    """
    Splits dataframes into x (containing the data)
    and y (containing the respective labels)
    :param dataframes: 4 dataframes read by read_csv()
    :param needs_embed: boolean if you want to embed the text
    -> text should be embedded if the project does not include "embedded_matrix",
    "embedded_text" and "embedded_text_test" files.
    To embed, "wiki-news-300d-1M.vec" should be included in the project
    """
    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = dataframes[0]['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x_lstm = dataframes[0].drop(columns=['dementia'])
    x_fe = dataframes[1].drop(columns=['dementia'])

    x_test = dataframes[2].drop(columns=['dementia'])
    y_test = dataframes[2]['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    x_test_lstm = dataframes[3].drop(columns=['dementia'])

    embed(x_lstm, y, x_fe, x_test, x_test_lstm, y_test, needs_embed)


def embed(x_lstm, y, x_fe, x_test, x_test_lstm, y_test, needs_embed):
    """
    Embeds the text in the training and test set using the wiki-news pre-trained vectors.
    :param x_lstm: training set text
    :param y: training set labels
    :param x_fe: training set features
    :param x_test: test set features
    :param x_test_lstm: test set text
    :param y_test: test set labels
    :param needs_embed: boolean if you want to embed the text
    -> text should be embedded if the project does not include "embedded_matrix",
    "embedded_text" and "embedded_text_test" files.
    To embed, "wiki-news-300d-1M.vec" should be included in the project
    """
    max_len = 197

    if needs_embed:
        #for train and validation set
        word2index, embedding_matrix = load_embeddings('wiki-news-300d-1M.vec', embedding_dim=300)

        out_matrix = []

        for text in x_lstm['text'].tolist():
            indices = []
            for w in text_to_word_sequence(text):
                indices.append(word2index[re.sub(r'[^\w\s]', '', w)])
            if len(indices) > max_len:
                max_len = len(indices)
            out_matrix.append(indices)

        encoded_texts = out_matrix

        padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

        store_data(padded_texts, 'embedded_text')
        store_data(embedding_matrix, 'embedded_matrix')


        # for test set
        word2index, embedding_matrix = load_embeddings('wiki-news-300d-1M.vec', embedding_dim=300)

        out_matrix = []

        for text in x_test_lstm['text'].tolist():
            indices = []
            for w in text_to_word_sequence(text):
                # Scotfree is present in the data, but it is not in wiki-news and
                # throws error
                if w == 'scotfree':
                    continue
                indices.append(word2index[re.sub(r'[^\w\s]', '', w)])
            if len(indices) > max_len:
                max_len = len(indices)
            out_matrix.append(indices)

        encoded_texts = out_matrix
        padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')
        store_data(padded_texts, 'embedded_text_test')

    embedding_matrix = load_data('embedded_matrix')
    padded_texts = load_data('embedded_text')
    embedding_matrix_test = load_data('embedded_text_test')
    for idx, el in enumerate(padded_texts):
        dataframes[0]['text'][idx] = el

    for idx, el in enumerate(embedding_matrix_test):
        dataframes[3]['text'][idx] = el
    x_test_lstm = dataframes[3]['text']
    do_kfold_validation(x_fe, y, embedding_matrix, max_len, x_test, x_test_lstm, y_test)


def store_data(padded_texts, file_name):
    """
    Stores embedded text into pickle for faster future loading
    :param padded_texts: embedded text
    :param file_name: filename to save file
    """
    pickle_file = open(file_name, 'ab')

    pickle.dump(padded_texts, pickle_file)
    pickle_file.close()


def load_data(file_name):
    """
    Loads pickle file
    :param file_name: name of the file to load
    :return: Embedded text read from the pickle file
    """
    pickle_file = open(file_name, 'rb')
    padded_texts = pickle.load(pickle_file)

    pickle_file.close()
    return padded_texts


def do_kfold_validation(x_fe, y, embedding_matrix, max_len, x_test, x_test_lstm, y_test):
    """
    Perform kfold cross validation to avoid overfitting
    :param x_fe: training set features
    :param y: training set labels
    :param embedding_matrix: embedded text matrix
    :param padded_texts: embedded text
    :param max_len: length of longest sentence
    :param x_test: test set features
    :param x_test_lstm: test set text
    :param y_test: test set labels
    """
    # initializes kfold with 5 folds, including shuffling,
    # using 7 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=9, shuffle=True)
    scaler = StandardScaler()


    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(dataframes[0]['text']):
        x_train_lstm, x_val_train = dataframes[0]['text'].loc[train_index], dataframes[0]['text'].loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        x_train_features, x_val_features = x_fe.loc[train_index], x_fe.loc[val_index]

    x_train_features = scaler.fit_transform(x_train_features)
    x_val_features = scaler.transform(x_val_features)
    x_test = scaler.fit_transform(x_test)

    train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val_train, y_val, x_train_features, x_val_features,
                x_test, x_test_lstm, y_test)


def train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val, y_val, x_t_f, x_v_f,
                x_test, x_test_lstm, y_test):
    """
    Creates and trains full architecture including fully-connected
    model and LSTM model
    """

    x_df = x_train_lstm.values
    y_df = x_val.values
    x_df_test = x_test_lstm.values
    a = np.vstack(x_df)

    b = np.vstack(y_df)

    c = np.vstack(x_df_test)

    fe_model = feature_layer()
    lstm = lstm_model(embedding_matrix, max_len)
    merged_layers = concatenate([fe_model.output, lstm.output])

    x = Dense(84, activation='tanh')(merged_layers)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    out = Activation('sigmoid')(x)
    model = Model([fe_model.input, lstm.input], [out])
    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
    es = EarlyStopping(monitor='val_acc', patience=100)
    history = model.fit([x_t_f, a], y_train, epochs=52,  verbose=1, validation_data=([x_v_f, b],  y_val), batch_size=a.shape[0], callbacks=[es])  # Fit the model
    evaluate_model(model, x_t_f, a, y_train, 'train')
    evaluate_model(model, x_v_f, b, y_val, 'validation')
    evaluate_model(model, x_test, c, y_test, 'test')
    y_pred = model.predict([x_t_f, a])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_train, y_pred, 'train')
    y_pred = model.predict([x_v_f, b])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_val, y_pred, 'validation')
    y_pred = model.predict([x_test, c])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_test, y_pred, 'test')

    plot_history(history)


def feature_layer():
    """
    Creates a fully-connected layer for features
    :return: Dense model
    """
    fe_model = Sequential()
    # input layer: 24 neurons just like the number of features
    fe_model.add(Dense(24, input_dim=24))
    return fe_model


def lstm_model(embedding_matrix, max_len):
    """
    Defines and returns the LSTM architecture for text
    :param embedding_matrix: matrix of embedded text
    :param max_len: max length of sentence
    :return: LSTM model
    """
    model = Sequential()

    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=max_len,
                        weights=[embedding_matrix],
                        trainable=False,
                        name='embedding_layer'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    return model


def evaluate_model(model, x_train_features, x_train_lstm, y_train, set):
    """
    Evaluates accuracy of the model
    :param model: trained model
    :param x_train_features: training set features
    :param x_train_lstm: training set text
    :param y_train: training set labels
    :param set: name of the evaluated set
    """
    # evaluate the model
    scores = model.evaluate([x_train_features, x_train_lstm], y_train)
    print("{0:<35s} {1:6.2f}%".format('Accuracy on ' + set + ' set:', scores[1] * 100))


def evaluate_sen_spec(y_true, y_pred, set):
    """
    Evaluates sensitivity and specificity
    on given dataset
    :param y_true: correct labels
    :param y_pred: predicted labels
    :param set: name of the evaluated dataset
    """
    # gets number of true negatives (tn), false positives (fp),
    # false negatives (fn) and true positives (tp)
    # by matching the predicted labels against the correct ones
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("{0:<35s} {1:6.2f}%".format('Specificity on ' + set + ' set:', specificity * 100))
    print("{0:<35s} {1:6.2f}%".format('Sensitivity on ' + set + ' set:', sensitivity * 100))


def transform_predictions(y_pred):
    """
    Scales each prediction > 0.5 to 1
    and <= 0.5 to 0 to get binary values
    :param y_pred:
    :return: scaled predictions
    """
    for i, v in enumerate(y_pred):
        if v > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    return y_pred


def plot_history(history):
    """
    Plots training history
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss


dataframes = read_csv()
split_dataframe(dataframes, False)
