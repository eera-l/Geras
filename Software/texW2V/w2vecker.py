import gensim
import os
from random import randint

model = gensim.models.KeyedVectors.load_word2vec_format('/home/federica/Documents/Thesis/Datasets/GoogleNews-vectors-negative300.bin', binary=True)
print(model.distance('man', 'woman'))

path = '/home/federica/Documents/Thesis/Programs/TextParser/dementia/dementia_no_functional_clean/'
pathControl = '/home/federica/Documents/Thesis/control/control_description'

files = []

for r, d, f, in os.walk(path):
    for file in f:
        if '.cha' in file:
            files.append(os.path.join(r, file))

words = []
lines = []

distances = []
total = []
randoms = []

randomDistances = []

control = open(pathControl, "r")
contentsControl = control.read()
data = contentsControl.split(" ")
data[len(data) - 1] = "hedge"

for file in files:
    file = open(file, "r")
    contents = file.read()
    dataFile = contents.split("\n")

    for df in dataFile:
        df = df.replace("_", " ")
        if not df:
            continue
        dataSpace = df.split(" ")
        for ds in dataSpace:
            if not ds or ds == "of" or ds == "a":
                continue
            words.append(ds)

    for w in words:
        for ww in data:
            distances.append(model.similarity(w, ww))

    words.clear()
    for i in range(0, 50):
        num = randint(0, distances(total) - 1)
        while num in randoms:
            num = randint(0, distances(total) - 1)
        randoms.append(num)

    #for i in randoms:
        #something

    average = sum(distances) / len(distances)
    print(average)
    distances.clear()
    total.append(average)

for i in range(0, 50):
    randomDistances.append(total[randoms[i]])

print("============================")
print(sum(randomDistances) / len(randomDistances))








































