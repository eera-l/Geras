import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/federica/Documents/Thesis/Programs/TextParser/control/control_features/'

files = []
rows = []

for r, d, f in os.walk(path):
    for file in f:
        if '.cha' in file:
            files.append(os.path.join(r, file))


for file in files:
    file = open(file, "r")
    lines = file.read()
    words = lines.split("\n")[:-1]
    rows.append(words)
    print(rows)

df = pd.DataFrame(np.array(rows), columns=['utterances', 'predicates', 'sentences', 'unique_words', 'utterances_length',
                                           'subordinates', 'coordinates',
                                           'unintelligibles', 'hesitations', 'incompletions', 'long_syllables',
                                           'pauses_syllables',
                                           'pauses', 'trailing_offs', 'overlaps', 'repetitions', 'retracing_correction',
                                           'retracing_reform', 'type_token_ratio', 'adverbs_ratio',
                                           'adjectives_ratio', 'idea_density'])
#df['hesitations'].hist()
#plt.show()
print(df)

