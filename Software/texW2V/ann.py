import numpy
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, regularizers, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

df = pd.read_csv('train_set', sep=',')

#Process data
df = shuffle(df)
y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
x = df.drop(columns=['dementia'])

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
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
# x_val = scaler.transform(x_val)



# create model
model = Sequential()
model.add(Dense(24, input_dim=24, activation='relu'))
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
history = model.fit(x_train, y_train, validation_split=0.2, epochs=1000, callbacks=[es])


# evaluate the model
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(history.history['val_acc'][-1])
# scores = model.evaluate(x_val, y_val)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(history.history.keys())
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

