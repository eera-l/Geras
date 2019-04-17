from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import pandas as pd

#Read data
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

#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_val = scaler.transform(x_val)


#training of the model
model = GaussianNB()
model.fit(x_train, y_train)

#evaluation
score = model.score(x_train, y_train)

print(score)

score = model.score(x_val, y_val)

print(score)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = plt.matshow(x.corr())
# ax.set_xticklabels([''])
# ax.set_yticklabels([''])
#
# fig.colorbar(cax)
# plt.show()