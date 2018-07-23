#coding:utf-8
import word2vec

word2vec.word2vec('self_train.txt', 'embedding.bin', size=100, verbose=True, min_count=0, window=5)
model = word2vec.load('embedding.bin')
print(model.vocab)
indexes, metrics = model.cosine('英国')
print (model.vocab[indexes])