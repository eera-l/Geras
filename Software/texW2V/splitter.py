import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

pathControl = '/home/federica/Documents/Thesis/Programs/TextParser/control/control_features/'
pathDementia = '/home/federica/Documents/Thesis/Programs/TextParser/dementia/dementia_features'

filesC = []
filesD = []
rows = []

for r, d, f in os.walk(pathControl):
    for file in f:
        if '.cha' in file:
            filesC.append(os.path.join(r, file))

for r, d, f in os.walk(pathDementia):
    for file in f:
        if '.cha' in file:
            filesD.append(os.path.join(r, file))


for file in filesC:
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    rows.append(words)

for file in filesD:
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    rows.append(words)

df = pd.DataFrame(np.array(rows), columns=['utterances', 'predicates', 'sentences', 'unique_words', 'utterances_length',
                                           'subordinates', 'coordinates',
                                           'unintelligibles', 'hesitations', 'incompletions', 'long_syllables',
                                           'pauses_syllables',
                                           'pauses', 'trailing_offs', 'overlaps', 'repetitions', 'retracing_correction',
                                           'retracing_reform', 'type_token_ratio', 'adverbs_ratio',
                                           'adjectives_ratio', 'idea_density'])
df['utterances'].hist()
plt.show()
#df.to_csv('dataset', index=False)
print(df)

