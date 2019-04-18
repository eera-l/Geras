import pandas as pd
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.utils import shuffle

embed_dim = 128
lstm_out = 200
batch_size = 32

df = pd.read_csv('train_set', sep=',')

#Process data
df = shuffle(df)
y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
x = df.drop(columns=['dementia'])

split_rate = 0.2
split_index = int(split_rate * df.shape[0])
x_val = x[:split_index]
y_val = y[:split_index]

x_train = x[split_index:]
y_train = y[split_index:]

Y = pd.get_dummies(x_train).values


model = Sequential()
#model.add(Embedding(2500, embed_dim,input_length = x_train.shape[1], dropout = 0.2))
model.add(LSTM(2, dropout_U = 0.2, dropout_W = 0.2, input_shape=(376, 23)))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


model.fit(x_train, y_train, batch_size =batch_size, nb_epoch = 1, verbose = 5)

