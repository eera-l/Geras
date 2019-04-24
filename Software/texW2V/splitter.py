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

#206 control texts
for i in range(0, 206):
    file = filesC[i]
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    words.append('N')
    rows.append(words)

#263 dementia texts
for i in range(0, 263):
    file = filesD[i]
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    words.append('Y')
    rows.append(words)

df = pd.DataFrame(np.array(rows), columns=['utterances', 'predicates', 'sentences', 'unique_words', 'utterances_length',
                                           'subordinates', 'coordinates',
                                           'unintelligibles', 'hesitations', 'incompletions', 'long_syllables',
                                           'pauses_syllables',
                                           'pauses', 'trailing_offs', 'overlaps', 'repetitions', 'retracing_correction',
                                           'retracing_reform', 'type_token_ratio', 'adverbs_ratio',
                                           'adjectives_ratio', 'idea_density', 'w2v_distance', 'age', 'dementia'])

df.to_csv('train_set', index=False)


rows.clear()

for i in range(206, len(filesC)):
    file = filesC[i]
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    words.append('N')
    rows.append(words)

for i in range(263, len(filesD)):
    file = filesD[i]
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    words.append('Y')
    rows.append(words)

df = pd.DataFrame(np.array(rows), columns=['utterances', 'predicates', 'sentences', 'unique_words', 'utterances_length',
                                           'subordinates', 'coordinates',
                                           'unintelligibles', 'hesitations', 'incompletions', 'long_syllables',
                                           'pauses_syllables',
                                           'pauses', 'trailing_offs', 'overlaps', 'repetitions', 'retracing_correction',
                                           'retracing_reform', 'type_token_ratio', 'adverbs_ratio',
                                           'adjectives_ratio', 'idea_density', 'w2v_distance', 'age', 'dementia'])
df.to_csv('test_set', index=False)

