import pandas as pd
import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers.embeddings import Embedding

"""
Reads data from .csv file
"""


def read_csv():
    global df

    # reads data and stores it into dataframe
    df = pd.read_csv('train_set_lstm', sep=',')


"""
Splits dataframe into x (containing the data)
and y (containing the respective labels)
"""


def split_dataframe():
    global df, x, y

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia'])
    x = x.iloc[1:]


"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation():
    # initializes kfold with 5 folds, including shuffling,
    # using 7 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=7, shuffle=True)

    global x_train, x_val, y_train, y_val

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]


def tokenize():
    global x_train, x_val, vocabulary_size, max_length, x_train_pad, x_val_pad, x_train_tokens, x_val_tokens
    tokenizer = Tokenizer()
    total_texts = x_train + x_val
    tokenizer.fit_on_texts(total_texts)

    max_length = max([len(s.split()) for s in total_texts])

    vocabulary_size = len(tokenizer.word_index) + 1

    x_train_tokens = tokenizer.texts_to_sequences(x_train)
    x_val_tokens = tokenizer.texts_to_sequences(x_val)

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
    x_val_pad = pad_sequences(x_val_tokens, maxlen=max_length, padding='post')


def train_model():
    global vocabulary_size, max_length, x_train_pad, y_train, x_val_pad, y_val, x_train_token, x_val_tokens
    model = Sequential()
    model.add(Embedding(vocabulary_size, 100, input_length=max_length))
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train_tokens, y_train, batch_size=128, epochs=50, validation_data=(x_val_tokens, y_val), verbose=2)
    evaluate_model(model, history)


"""
Evaluates accuracy of the model
"""


def evaluate_model(model, history):
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', scores[1] * 100))
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', float(history.history['val_acc'][-1]) * 100))


read_csv()
split_dataframe()
do_kfold_validation()
tokenize()
train_model()