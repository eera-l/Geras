"""
bayesian_network.py
author: Federica Comuni
Implements a Bayesian Network with Gaussian Naive Bayes classifier
reads raw data from train_set.csv and converts it to a dataframe
then splits the dataset into validation and train set
performs 5-fold cross validation
trains the model on the train set and evaluates the classification accuracy on train set and validation set
evaluates sensitivity and specificity on both datasets
plots a graph of the correlation between features
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


"""
Reads data from .csv file
"""


def read_csv():

    # reads data and stores it into dataframe
    df = pd.read_csv('train_set', sep=',')
    df_test = pd.read_csv('test_set', sep=',')
    split_dataframe(df, df_test)



"""
Splits dataframe into x (containing the data)
and y (containing the respective labels)
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

    # uses a Gaussian Naive Bayes classifier
    model = GaussianNB()

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        # training of the model
        model.fit(x_train, y_train)

    evaluate_accuracy(model, x_train, x_val, x_test, y_train, y_val, y_test)
    # plot_correlation(x)
    plot_frequency(x)


"""
Evaluates accuracy on train set
and validation set
"""


def evaluate_accuracy(model, x_train, x_val, x_test, y_train, y_val, y_test):
    # evaluation on train set
    score = model.score(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', score * 100))

    # evaluation on validation set
    score = model.score(x_val, y_val)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', score * 100))

    # evaluation on test set
    score = model.score(x_test, y_test)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on test set:', score * 100))

    # stores predicted labels for the train set
    # in an array
    y_pred = model.predict(x_train)
    evaluate_sen_spec(y_train, y_pred, 'train')

    # stores predicted labels for the validation set
    # in an array
    y_pred = model.predict(x_val)
    evaluate_sen_spec(y_val, y_pred, 'validation')

    # stores predicted labels for the test set
    # in an array
    y_pred = model.predict(x_test)
    evaluate_sen_spec(y_test, y_pred, 'test')


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
plots graph of correlation between features using matplotlib
"""


def plot_correlation(x):
    labels = x.columns.values
    fig = plt.figure(num='Features correlation')
    ax = fig.add_subplot(111)
    cax = ax.matshow(x.corr())
    fig.colorbar(cax)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks(range(len(labels)), labels)
    plt.show()


def plot_frequency(x):
    plt.hist(x['hesitations'])
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    plt.show()


read_csv()
