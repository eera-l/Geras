from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

#Read data
df = pd.read_csv('train_set', sep=',')

#Process data
df = shuffle(df)
y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])


split_rate = 0.2
split_index = int(split_rate * df.shape[0])
x_val = x[:split_index]
y_val = y[:split_index]

x_train = x[split_index:]
y_train = y[split_index:]

#attempt at normalization
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_val = scaler.transform(x_val)


#training of the model
model = GaussianNB()
model.fit(x_train, y_train)

for i in range(0, 6):
    df = shuffle(df)
    y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
    x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])

    x_train = x[split_index:]
    y_train = y[split_index:]

    x_val = x[:split_index]
    y_val = y[:split_index]

    model.fit(x_train, y_train)


#evaluation
score = model.score(x_train, y_train)

print(score)

score = model.score(x_val, y_val)

print(score)

# labels = x.columns.values
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(x.corr())
# fig.colorbar(cax)
# plt.xticks(range(len(labels)), labels, rotation='vertical')
# plt.yticks(range(len(labels)), labels)
# plt.show()
