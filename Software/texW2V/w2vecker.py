import gensim
import os
from random import randint
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

#model = gensim.models.KeyedVectors.load_word2vec_format('/home/federica/Documents/Thesis/Datasets/GoogleNews-vectors-negative300.bin', binary=True)
top_directory = "/home/federica/Documents/Thesis/Programs/TextParser/corpus.txt"

#corpus = api.load("/home/federica/Documents/Thesis/Programs/TextParser/corpus.txt")



# def iter_documents(top_directory):
#     """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
#     document = open(top_directory).read()  # read the entire document, as one big string
#     document = document.replace("\n", " ")
#     yield gensim.utils.tokenize(document, lower=True) # or whatever tokenization suits you
#
#
# class MyCorpus(object):
#     def __init__(self, top_dir):
#         self.top_dir = top_dir
#         self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir))
#         self.dictionary.filter_extremes(no_below=1, keep_n=30000) # check API docs for pruning params
#
#     def __iter__(self):
#         for tokens in iter_documents(self.top_dir):
#             yield self.dictionary.doc2bow(tokens)
#
#
# corpus = MyCorpus(top_directory)  # create a dictionary
# for vector in corpus:  # convert each document to a bag-of-word vector
#     print(vector)

model = api.load('fasttext-wiki-news-subwords-300')
#model = Word2Vec(corpus)
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

# for i in range(0, 50):
    #     num = randint(0, distances(total) - 1)
    #     while num in randoms:
    #         num = randint(0, distances(total) - 1)
    #     randoms.append(num)

# for i in range(0, 50):
#     randomDistances.append(total[randoms[i]])

print("============================")
print(sum(total.values()) / len(total))








































