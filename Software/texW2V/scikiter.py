from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

#Read data
df = pd.read_csv('train_set', sep=',')

#Process data
df = shuffle(df)
y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
x = df.drop(columns=['dementia'])

#TODO split x and y into 80-20
split_rate = 0.2
split_index = int(split_rate * df.shape[0])
x_val = x[:split_index]
y_val = y[:split_index]

x_train = x[split_index:]
y_train = y[split_index:]




#valid = pd.read_csv('validation_set', sep=',')

#training of the model
model = BernoulliNB()
model.fit(x_train, y_train)

#evaluation
score = model.score(x_train, y_train)

#predicted = model.predict(valid)

print(score)

score = model.score(x_val, y_val)

print(score)