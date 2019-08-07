"""
dense_neural_network.py
author: Federica Comuni
Implements a 4-layer artificial neural network using Keras
reads raw data from train_set.csv and converts it to a dataframe
then splits the dataset into validation and train set
performs 5-fold cross validation and normalizes the features
trains the model and evaluates accuracy, specificity and sensitivity on train set and validation set
plots accuracy and loss history of both sets
"""

import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


"""
Reads data and stores it into dataframe df
"""


def read_data():
    df = pd.read_csv('train_set', sep=',')
    df_test = pd.read_csv('test_set', sep=',')
    np.random.seed(21)
    split_dataframe(df, df_test)



"""
Splits dataframe into dataframe x, containing data for all features
and y, containing corresponding label (healthy or dementia)
"""


def split_dataframe(df, df_test):

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])

    y_test = df_test['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    x_test = df_test.drop(columns=['dementia', 'pauses', 'retracing_reform'])
    do_kfold_validation(x, y, x_test, y_test)


"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation(x, y, x_test, y_test):
    # initializes kfold with 5 folds, including shuffling,
    # using 9 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=9, shuffle=True)

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        normalize(x_train, y_train, x_val, y_val, x_test, y_test)


"""
Normalizes datasets to ensure that all features
have mean = 0 and standard deviation = 1
"""


def normalize(x_train, y_train, x_val, y_val, x_test, y_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)
    x_test = scaler.fit_transform(x_test)
    train_model(x_train, y_train, x_val, y_val, x_test, y_test)


"""
Creates a 4-layer neural network and trains it
"""


def train_model(x_train, y_train, x_val, y_val, x_test, y_test):
    # create model
    model = Sequential()
    # input layer: 22 neurons just like the number of features
    model.add(Dense(22, input_dim=22, activation='relu'))
    # drops out to prevent overfitting
    model.add(Dropout(0.2))
    # first hidden layer
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    # second hidden layer
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # early stopping allows the model to stop training automatically
    # when the chosen value (validation set accuracy) reaches its peak and starts decreasing
    es = EarlyStopping(monitor='val_acc', patience=14)
    history = model.fit(x_train, y_train, epochs=100, callbacks=[es], validation_data=(x_val, y_val))
    evaluate_model(model, x_train, y_train, 'train')
    evaluate_model(model, x_val, y_val, 'validation')
    evaluate_model(model, x_test, y_test, 'test')
    y_pred = model.predict(x_train)
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_train, y_pred, 'train')
    y_pred = model.predict(x_val)
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_val, y_pred, 'validation')
    y_pred = model.predict(x_test)
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_test, y_pred, 'test')
    plot_history(history)


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
    print("{0:<35s} {1:6.2f}%".format('Specificity on ' + set + ' set:', specificity * 100))
    print("{0:<35s} {1:6.2f}%".format('Sensitivity on ' + set + ' set:', sensitivity * 100))


"""
Evaluates accuracy of the model
"""


def evaluate_model(model, x, y, set):
    # evaluate the model
    scores = model.evaluate(x, y)
    print("{0:<35s} {1:6.2f}%".format('Accuracy on ' + set + ' set:', scores[1] * 100))
    print("{0:<35s} {1:6.2f}%".format('Loss on ' + set + ' set:', scores[0] * 100))


"""
Plots history of accuracy and loss during epochs
"""


def plot_history(history):

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


read_data()

