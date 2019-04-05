import gensim
import os

model = gensim.models.KeyedVectors.load_word2vec_format('/home/federica/Documents/Thesis/Datasets/GoogleNews-vectors-negative300.bin', binary=True)
print(model.similarity('man', 'woman'))

path = '/home/federica/Documents/Thesis/Programs/TextParser/control_no_functional_clean/'
pathControl = '/home/federica/Documents/Thesis/control_description'

files = []

for r, d, f, in os.walk(path):
    for file in f:
        if '.cha' in file:
            files.append(os.path.join(r, file))

words = []
lines = []

distances = []

control = open(pathControl, "r")
contentsControl = control.read()
data = contentsControl.split(" ")
data[len(data) - 1] = "hedge"

for file in files:
    file = open(file, "r")
    contents = file.read()
    dataFile = contents.split("\n")

    for df in dataFile:
        if not df:
            continue
        dataSpace = df.split(" ")
        for ds in dataSpace:
            if not ds:
                continue
            words.append(ds)

    for w in words:
        for ww in data:
            distances.append(model.similarity(w, ww))


    average = sum(distances) / len(distances)
    print(average)

    distances.clear()






