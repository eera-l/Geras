import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('/home/federica/Documents/Thesis/Datasets/GoogleNews-vectors-negative300.bin', binary=True)
print(model.similarity('man', 'woman'))



