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
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Bidirectional, Concatenate, concatenate, \
    BatchNormalization, PReLU, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import StandardScaler

from load_glove_embeddings import load_glove_embeddings
import pickle
import matplotlib.pyplot as plt

"""
Reads data from .csv file
"""


def read_csv():

    # reads data and stores it into dataframe
    df = pd.read_csv('train_set_lstm_90', sep=',')
    df_2 = pd.read_csv('train_set_90', sep=',')
    dataframes = [df, df_2]
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
    embed(x_lstm, dataframes, y, x_fe)


def embed(x_lstm, df, y, x_fe):
    max_len = 197
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
    print(max_len)

    padded_texts = pad_sequences(encoded_texts, maxlen=max_len, padding='post')

    # for idx, el in enumerate(padded_texts):
    #     a = np.array(el)
    #     a = np.reshape(a, (70, 1))
    #     padded_texts[idx] = a

    # for idx, el in enumerate(df['text']):
    #     if df['text'][idx] != 'text':
    #         a = df['text'][idx]
    #         a = np.reshape(a, (70, 1))
    #        x_lstm[idx] = a
    store_data(padded_texts, 'embedded_text_90')
    padded_texts = load_data('embedded_text_90')
    store_data(embedding_matrix, 'embedded_matrix_90')
    embedding_matrix = load_data('embedded_matrix_90')
    for idx, el in enumerate(padded_texts):
        dataframes[0]['text'][idx] = el
    do_kfold_validation(x_lstm, x_fe, y, embedding_matrix, padded_texts, max_len)


def store_data(padded_texts, file_name):
    pickle_file = open(file_name, 'ab')

    pickle.dump(padded_texts, pickle_file)
    pickle_file.close()


def load_data(file_name):
    pickle_file = open(file_name, 'rb')
    padded_texts = pickle.load(pickle_file)

    pickle_file.close()
    return padded_texts


def normalize(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x

"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation(x_lstm, x_fe, y, embedding_matrix, padded_texts, max_len):
    # initializes kfold with 10 folds, including shuffling,
    # using 7 as seed for the shuffling
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(dataframes[0]['text']):
        x_train_lstm, x_val_train = dataframes[0]['text'].loc[train_index], dataframes[0]['text'].loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]

    for train_index, val_index in kfold.split(x_fe):
        x_train_features, x_val_features = x_fe.loc[train_index], x_fe.loc[val_index]

    x_train_features = normalize(x_train_features)
    x_val_features = normalize(x_val_features)

    train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val_train, y_val, padded_texts, y, x_train_features, x_val_features)


def train_model(x_train_lstm, y_train, embedding_matrix, max_len, x_val, y_val, padded_texts, y, x_t_f, x_v_f):

    # for idx, el in enumerate(x_train):
    #     if el == 'text':
    #         x_train.pop(0)
    #     a = el
    #     a = np.reshape(a, (70, 1))
    #     x_train[idx] = a

    # x_train.pop(0)


    x_df = x_train_lstm.values
    y_df = x_val.values

    # a = np.zeros([353, 70])
    a = np.vstack(x_df)

    # b = np.zeros([88, 70])
    b = np.vstack(y_df)

    fe_model = feature_layer()
    lstm = lstm_model(embedding_matrix, max_len)
    merged_layers = concatenate([fe_model.output, lstm.output])
    # model.add(Concatenate([fe_model, lstm]))
    # model.add(Dense(1,activation='sigmoid'))

    x = Dense(84, activation='relu')(merged_layers)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    out = Activation('sigmoid')(x)
    model = Model([fe_model.input, lstm.input], [out])
    print(model.summary())
    '''model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))'''

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
    # print(model.summary())  # Summarize the model
    es = EarlyStopping(monitor='val_acc', patience=100)
    history = model.fit([x_t_f, a], y_train, epochs=400,  verbose=1, validation_data=([x_v_f, b],  y_val), batch_size=a.shape[0], callbacks=[es])  # Fit the model
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


dataframes = read_csv()
split_dataframe(dataframes)
# do_kfold_validation()
