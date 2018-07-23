import codecs

import jieba
import re
hanzi = re.compile(r"[\u4e00-\u9fa5]+")
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
def get_word_index(train_data):
    w_idx = {".":0}
    for line in train_data:
        for w in line:
            if w not in w_idx:
                w_idx[w] = len(w_idx)
    idx_w = {i:j for i,j in w_idx.items()}

    return w_idx, idx_w

def get_train_data():
    data = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\pao_mo_zhi_xia.txt"

    for line in codecs.open(train_f, "r","utf-8"):
        line = [i.strip() for i in line.split() if i.strip()]
        for short_line in line:
            ans = []
            for w in jieba.cut(short_line.strip()):
                if hanzi.match(w):
                    ans.append(w)
            data.append(ans)
    return data

def get_tri_gram(train_data):
    ans =  []
    for test_sent in train_data:
        ans.extend( [((test_sent[i], test_sent[i + 1]), test_sent[i + 2])
         for i in range(len(test_sent) - 2)])
    return ans
