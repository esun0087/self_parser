import codecs
import re

CONTEXT_SIZE = 5  # 2 words to the left, 2 to the right
EMBEDDING_SIZE = 128
hanzi = re.compile(r"[\u4e00-\u9fa5]+")
def get_word_index(train_data):
    w_idx = {}
    for line in train_data:
        for w in line:
            if w not in w_idx:
                w_idx[w] = len(w_idx)
    idx_w = {j:i for i,j in w_idx.items()}

    return w_idx, idx_w
def get_train_data():
    data = []
    # train_f = r"D:\study\nlp\self_parser\average_perceptron\data\zhihu.txt"
    train_f = r"self_train.txt"
    for line in codecs.open(train_f, "r","utf-8"):
        line = [i.strip() for i in line.split() if i.strip()]
        data.append(line)
    print ("get train over", len(data))
    return data

def get_cbow_context(sent, c_s, pos):
    t = []
    for j in range(-c_s, c_s + 1):
        if pos + j >= 0 and pos + j < len(sent) and  j != 0:
            t.append(sent[pos + j])
    return t

def get_skip_context(sent, c_s, pos):
    t = []
    for j in range(-c_s, c_s + 1):
        if pos + j >= 0 and pos + j < len(sent) and  j != 0:
            t.append(sent[pos + j])
    return t
def get_cbow_gram(train_data):
    ans =  []
    for test_sent in train_data:
        for i in range(len(test_sent)):
            t = get_cbow_context(test_sent, CONTEXT_SIZE, i)
            ans.append((tuple(t), test_sent[i]))
    return ans


def get_skip_gram(train_data):
    ans =  []
    for test_sent in train_data:
        for i in range(len(test_sent)):
            t = get_skip_context(test_sent, CONTEXT_SIZE, i)
            for kk in t:
                ans.append(((test_sent[i],), kk))
    return ans