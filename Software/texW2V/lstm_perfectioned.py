import re

import pandas as pd
import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras_preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras_preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Bidirectional, Concatenate, concatenate, \
    BatchNormalization, PReLU, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import StandardScaler

from load_glove_embeddings import load_glove_embeddings
import pickle
import matplotlib.pyplot as plt
import time

"""
Reads data from .csv file
"""


def read_csv():
    # reads data and stores it into dataframe
    df = pd.read_csv('train_set_lstm', sep=',')
    df_2 = pd.read_csv('train_set', sep=',')
    df_test = pd.read_csv('test_set', sep=',')
    df_test_lstm = pd.read_csv('test_set_lstm', sep=',')
    dataframes = [df, df_2, df_test, df_test_lstm]
    np.random.seed(21)
    return dataframes


"""
Splits dataframe into x (containing the data)
and y (containing the respective labels)
"""


def split_dataframe(dataframes):

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

    embed(x_lstm, dataframes, y, x_fe, x_test, x_test_lstm, y_test)


def embed(x_lstm, df, y, x_fe, x_test, x_test_lstm, y_test):
    max_len = 197

    #for train and validation set
    word2index, embedding_matrix = load_glove_embeddings('wiki-news-300d-1M.vec', embedding_dim=300)

    out_matrix = []

    for text in x_lstm['text'].tolist():
        indices = []
        for w in text_to_word_sequence(text):
            indices.append(word2index[re.sub(r'[^\w\s]', '', w)])
        if len(indices) > max_len:
            max_len = len(indices)
        out_matrix.append(indices)

    encoded_texts = out_matrix
    # print(max_len)

    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

    store_data(padded_texts, 'embedded_text')
    store_data(embedding_matrix, 'embedded_matrix')


    # for test set
    word2index, embedding_matrix = load_glove_embeddings('wiki-news-300d-1M.vec', embedding_dim=300)

    out_matrix = []

    for text in x_test_lstm['text'].tolist():
        indices = []
        for w in text_to_word_sequence(text):
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
    do_kfold_validation(x_lstm, x_fe, y, embedding_matrix, padded_texts, max_len, x_test, x_test_lstm, y_test)


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


def do_kfold_validation(x_lstm, x_fe, y, embedding_matrix, padded_texts, max_len, x_test, x_test_lstm, y_test):
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


    train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val_train, y_val, padded_texts, y, x_train_features, x_val_features,
                x_test, x_test_lstm, y_test)


def train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val, y_val, padded_texts, y, x_t_f, x_v_f,
                x_test, x_test_lstm, y_test):

    # for idx, el in enumerate(x_train):
    #     if el == 'text':
    #         x_train.pop(0)
    #     a = el
    #     a = np.reshape(a, (70, 1))
    #     x_train[idx] = a

    # x_train.pop(0)


    x_df = x_train_lstm.values
    y_df = x_val.values
    x_df_test = x_test_lstm.values

    # a = np.zeros([353, 70])
    a = np.vstack(x_df)

    # b = np.zeros([88, 70])
    b = np.vstack(y_df)

    c = np.vstack(x_df_test)

    fe_model = feature_layer()
    lstm = lstm_model(embedding_matrix, max_len)
    merged_layers = concatenate([fe_model.output, lstm.output])
    # model.add(Concatenate([fe_model, lstm]))
    # model.add(Dense(1,activation='sigmoid'))

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
    # print(model.summary())  # Summarize the model
    es = EarlyStopping(monitor='val_acc', patience=100)
    history = model.fit([x_t_f, a], y_train, epochs=52,  verbose=1, validation_data=([x_v_f, b],  y_val), batch_size=a.shape[0], callbacks=[es])  # Fit the model
    evaluate_model(model, history, x_t_f, a, y_train, 'train')
    evaluate_model(model, history, x_v_f, b, y_val, 'validation')
    evaluate_model(model, history, x_test, c, y_test, 'test')
    y_pred = model.predict([x_t_f, a])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_train, y_pred, 'train')
    y_pred = model.predict([x_v_f, b])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_val, y_pred, 'validation')
    y_pred = model.predict([x_test, c])
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_test, y_pred, 'test')

    # history = model.fit(padded_texts, y, batch_size=padded_texts.shape[0], epochs=3000, verbose=1, callbacks=[es])  # Fit the model
    # loss, accuracy = model.evaluate(a, y_train, verbose=0)  # Evaluate the model
    # print('Train set accuracy: %0.3f' % accuracy)
    # loss, accuracy = model.evaluate(b, y_val, verbose=0)
    # print('Validation set accuracy: %0.3f' % accuracy)
    plot_history(history)


def feature_layer():
    fe_model = Sequential()
    # input layer: 22 neurons just like the number of features
    fe_model.add(Dense(24, input_dim=24))
    return fe_model


def lstm_model(embedding_matrix, max_len):
    model = Sequential()
    # embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
    #                             output_dim=embedding_matrix.shape[1],
    #                             input_length=max_len,
    #                             weights=[embedding_matrix],
    #                             trainable=False,
    #                             name='embedding_layer')
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=max_len,
                        weights=[embedding_matrix],
                        trainable=False,
                        name='embedding_layer'))
    model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    return model


"""
Evaluates accuracy of the model
"""


def evaluate_model(model, history, x_train_features, x_train_lstm, y_train, set):
    # evaluate the model
    scores = model.evaluate([x_train_features, x_train_lstm], y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on ' + set + ' set:', scores[1] * 100))

"""
Evaluates sensitivity and specificity
on given dataset
"""


def evaluate_sen_spec(y_true, y_pred, set):
    # gets number of true negatives (tn), false positives (fp),
    # false negatives (fn) and true positives (tp)
    # by matching the predicted labels against the correct ones
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("{0:<35s} {1:6.3f}%".format('Specificity on ' + set + ' set:', specificity))
    print("{0:<35s} {1:6.3f}%".format('Sensitivity on ' + set + ' set:', sensitivity))


"""
Scales each prediction > 0.5 to 1 
and <= 0.5 to 0 to get binary values
"""


def transform_predictions(y_pred):
    for i, v in enumerate(y_pred):
        if v > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    return y_pred


def plot_history(history):
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
split_dataframe(dataframes)
# do_kfold_validation()
