from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

def read_csv():
    #Read data
    global df
    df = pd.read_csv('train_set', sep=',')


def split_dataframe():
    #Process data
    global df, x, y
    df = shuffle(df)
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform', 'type_token_ratio'])


def do_kfold_validation():
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    global model
    model = GaussianNB()
    global x_train, x_val, y_train, y_val

    for train_index, val_index in kfold.split(x):
        x_train, x_val = x.ix[train_index], x.ix[val_index]
        y_train, y_val = y.ix[train_index], y.ix[val_index]
        # training of the model
        model.fit(x_train, y_train)


def evaluate_accuracy():
    #evaluation
    score = model.score(x_train, y_train)
    print(score)
    score = model.score(x_val, y_val)
    print(score)


def plot_correlation():
    labels = x.columns.values
    fig = plt.figure()
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