# scikiter.py
# author: Federica Comuni
# Implements a Bayesian Network with Gaussian Naive Bayes classifier
# reads raw data from train_set.csv and converts it to a dataframe
# then splits the dataset into validation and train set
# performs 10-fold cross validation
# trains the model on the train set and evaluates the classification accuracy on train set and validation set
# evaluates sensitivity and specificity on both datasets
# plots a graph of the correlation between features

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


def read_csv():
    global df

    # read data and stores it into dataframe
    df = pd.read_csv('train_set', sep=',')


def split_dataframe():
    # Process data
    global df, x, y

    # take only dementia column (which are the labels, Y for dementia and N for control)
    # and convert to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drop columns with empty values (pauses and retracing_reform)
    # and drop dementia column because it is not needed
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])


# Perform kfold cross validation to avoid overfitting
def do_kfold_validation():
    # initializes kfold with 10 folds, including shuffling,
    # using 42 as seed for the shuffling
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)

    global model, x_train, x_val, y_train, y_val

    # uses a Gaussian Naive Bayes classifier
    model = GaussianNB()

    # splits datasets into folds and trains the model
    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.loc[train_index], x.loc[val_index]
        y_train, y_val = y.loc[train_index], y.loc[val_index]
        # training of the model
        model.fit(x_train, y_train)


def evaluate_accuracy():
    # evaluation on train set
    score = model.score(x_train, y_train)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on train set:', score * 100))

    # evaluation on validation set
    score = model.score(x_val, y_val)
    print("{0:<35s} {1:6.3f}%".format('Accuracy on validation set:', score * 100))

    # stores predicted labels for the train set
    # in an array
    y_pred = model.predict(x_train)
    evaluate_sen_spec(y_train, y_pred, 'train')

    # stores predicted labels for the validation set
    # in an array
    y_pred = model.predict(x_val)
    evaluate_sen_spec(y_val, y_pred, 'validation')


def evaluate_sen_spec(y_true, y_pred, set):
    # gets number of true negatives (tn), false positives (fp),
    # false negatives (fn) and true positives (tp) by
    # matching the predicted labels against the correct ones
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print("{0:<35s} {1:6.3f}%".format('Specificity on ' + set + ' set:', specificity))
    print("{0:<35s} {1:6.3f}%".format('Sensitivity on ' + set + ' set:', sensitivity))


# plots graph of correlation between features using matplotlib
def plot_correlation():
    labels = x.columns.values
    fig = plt.figure(num='Features correlation')
    ax = fig.add_subplot(111)
    cax = ax.matshow(x.corr())
    fig.colorbar(cax)
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.yticks(range(len(labels)), labels)
    plt.show()


read_csv()
split_dataframe()
do_kfold_validation()
evaluate_accuracy()
plot_correlation()
