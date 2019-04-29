import numpy
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, regularizers, Dropout
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_data():
    global df
    df = pd.read_csv('train_set', sep=',')


def split_dataframe():
    global df, x, y, x_train, y_train

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])
    x_train = x
    y_train = y

# Perform kfold cross validation to avoid overfitting
def do_kfold_validation():
    global x_train, y_train, split_rate
    numpy.random.seed(7)
    split_rate = 0.2


def normalize():
    global x_train
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)


def train_model():
    global model, history
    # create model
    model = Sequential()
    model.add(Dense(22, input_dim=22, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model

    es = EarlyStopping(monitor='val_acc', patience=14)
    history = model.fit(x_train, y_train, validation_split=split_rate, epochs=100, callbacks=[es])
    print(history.history['val_acc'])


def evaluate_model():
    # evaluate the model
    scores = model.evaluate(x_train, y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print(history.history['val_acc'][-1])

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
evaluate_model()

