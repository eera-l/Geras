"""
ann.py
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


"""
Reads data and stores it into dataframe df
"""


def read_data():
    global df
    df = pd.read_csv('train_set', sep=',')



"""
Splits dataframe into dataframe x, containing data for all features
and y, containing corresponding label (healthy or dementia)
"""


def split_dataframe():
    global df, x, y, x_train, y_train

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])


"""
Perform kfold cross validation to avoid overfitting
"""


def do_kfold_validation():
    # initializes kfold with 5 folds, including shuffling,
    # using 9 as seed for the shuffling
    kfold = KFold(n_splits=5, random_state=9, shuffle=True)

    global x_train, x_val, y_train, y_val

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]


"""
Normalizes datasets to ensure that all features
have mean = 0 and standard deviation = 1
"""


def normalize():
    global x_train, x_val
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.fit_transform(x_val)


"""
Creates a 4-layer neural network and trains it
"""


def train_model():
    global model, history
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
    evaluate_model()
    y_pred = model.predict(x_train)
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_train, y_pred, 'train')
    y_pred = model.predict(x_val)
    y_pred = transform_predictions(y_pred)
    evaluate_sen_spec(y_val, y_pred, 'validation')


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
    print("{0:<35s} {1:6.3f}%".format('Specificity on ' + set + ' set:', specificity))
    print("{0:<35s} {1:6.3f}%".format('Sensitivity on ' + set + ' set:', sensitivity))


"""
Evaluates accuracy of the model
"""


def evaluate_model():
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', scores[1] * 100))
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', float(history.history['val_acc'][-1]) * 100))


"""
Plots history of accuracy and loss during epochs
"""


def plot_history():

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


read_data()
split_dataframe()
do_kfold_validation()
normalize()
train_model()
plot_history()

