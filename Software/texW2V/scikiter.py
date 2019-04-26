from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

#Read data
df = pd.read_csv('train_set', sep=',')

#Process data
df = shuffle(df)
y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
x = df.drop(columns=['dementia', 'pauses', 'retracing_reform', 'type_token_ratio'])

kfold = KFold(n_splits=10, random_state=42, shuffle=True)

model = GaussianNB()

for train_index, val_index in kfold.split(x):
    x_train, x_val = x.ix[train_index], x.ix[val_index]
    y_train, y_val = y.ix[train_index], y.ix[val_index]
    # training of the model

    model.fit(x_train, y_train)

# PROPER THING BEFORE KFOLD
# split_rate = 0.2
# split_index = int(split_rate * df.shape[0])
# x_val = x[:split_index]
# y_val = y[:split_index]
#
# x_train = x[split_index:]
# y_train = y[split_index:]


# KFOLD VALIDATION WITHOUT SCIKIT KFOLD
# for i in range(0, 11):
#     df = shuffle(df)
#     y = df['dementia'].apply(lambda x : 1 if x == 'Y' else 0)
#     x = df.drop(columns=['dementia', 'pauses', 'retracing_reform'])
#
#     x_train = x[split_index:]
#     y_train = y[split_index:]
#
#     x_val = x[:split_index]
#     y_val = y[:split_index]
#
#     model.fit(x_train, y_train)


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
