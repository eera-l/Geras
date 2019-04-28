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
    global df, x, y

    # takes only dementia column (which are the labels, Y for dementia and N for control)
    # and converts to numbers: 1 for Y and 0 for N
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)

    # drops columns with empty values (pauses and retracing_reform)
    # and drops dementia column because it is not needed
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])


# Perform kfold cross validation to avoid overfitting
def do_kfold_validation():
    global x_train, y_train, x_val, y_val
    numpy.random.seed(7)
    split_rate = 0.2
    split_index = int(split_rate * df.shape[0])
    # x_val = x[:split_index]
    # y_val = y[:split_index]
    #
    # x_train = x[split_index:]
    # y_train = y[split_index:]
    x_train = x
    y_train = y


def normalize():
    global x_train, x_val
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    # x_val = scaler.transform(x_val)

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

    es = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, callbacks=[es])


def evaluate_model():
    # evaluate the model
    loss, acc = model.evaluate(x_train, y_train)
    print(loss, acc)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # print(history.history['val_acc'][-1])
    # loss, acc = model.evaluate(x_val, y_val)
    # print(loss, acc)
    # print(history.history['val_acc'][-1])
    # print(scores)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # print(history.history.keys())
    # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


read_data()
split_dataframe()
do_kfold_validation()
normalize()
train_model()
evaluate_model()

