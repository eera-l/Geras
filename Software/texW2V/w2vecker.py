import os
import gensim.downloader as api

#Actually used Fasttext Wiki as pretrained dataset
#instead of Google News
model = api.load('fasttext-wiki-news-subwords-300')
print(model.most_similar("cat"))

path = '/home/federica/Documents/Thesis/Programs/TextParser/dementia/dementia_no_functional_clean/'
pathControl = '/home/federica/Documents/Thesis/control_description'

files = []

for r, d, f, in os.walk(path):
    for file in f:
        if '.cha' in file:
            files.append(os.path.join(r, file))

words = []
lines = []

distances = []
total = dict()
randoms = []

randomDistances = []

control = open(pathControl, "r")
contentsControl = control.read()
data = contentsControl.split(" ")
data[len(data) - 1] = "hedge"

for file in files:
    with open(file, "r") as f:
        contents = f.read()
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
            try:
                distances.append(model.similarity(w, ww))
            except KeyError as e:
                pass

    words.clear()

    average = sum(distances) / len(distances)
    print(average)
    distances.clear()
    total[os.path.basename(file)] = average

for file, average in total.items():
    with open("/home/federica/Documents/Thesis/Programs/TextParser/dementia/dementia_w2v_distance/" + file,
                "w") as singlefile:
        singlefile.write(str(average))

print("============================")
print(sum(total.values()) / len(total))








































