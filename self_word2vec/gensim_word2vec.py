#coding:utf-8
from gensim.models import word2vec

sentences = word2vec.LineSentence('self_train.txt')

model = word2vec.Word2Vec(sentences, hs=0, min_count=0, window=8, size=128, iter=1000)
req_count = 5
for key in model.wv.similar_by_word('羽毛球', topn =100):
    req_count -= 1
    print (key[0], key[1])
    if req_count == 0:
        break